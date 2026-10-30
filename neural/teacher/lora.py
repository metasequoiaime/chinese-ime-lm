"""LoRA fine-tuning of the teacher on the distillation pools with mlx-lm.

Every training example is one pool from `record.py` (`<data>/pools/train.jsonl`), read in the file's order (the pools are shuffled with a fixed seed, so any prefix is a random sample). The sequence is what `score_pool.py` scores: the start token (`eos`), the case's context tokenized on its own, then the gold tokenized on its own.

Two losses:

- `lm`: cross-entropy on the gold's tokens only, the context is not a target (mlx-lm's own loss with the prompt masked). This raises log P(gold | context), the score the teacher ranks by. Trained with `mlx_lm.tuner.trainer.train`.
- `rank`: the pool's candidates are scored the way `score_pool.py` scores them (sum of token log-probabilities after the same prefix, the prefix run once and its cache copied per candidate), and the loss is the cross-entropy of a softmax over those sums with the gold as the label, plus `--lm-weight` times the `lm` loss per gold token so the scores stay a language model's. This pushes the gold above the pool's negatives directly. One pool per micro-batch, `--grad-accumulation` pools per update, in a plain loop (the shapes change with every pool, so mlx-lm's compiled step would recompile every time).

C4 pages full of gambling and SEO text (`SPAM`, a word list; 14.8% of the C4 clauses in the previous step) are left out of training with `--drop-spam`.

The adapter directory gets `adapter_config.json` (what `mlx_lm.load(adapter_path=...)` reads), the final `adapters.safetensors`, a checkpoint every `--save-every` updates (`step-<n>/`, each loadable on its own), and `train-log.jsonl` with loss, learning rate, tokens per second, wall-clock and peak memory.

usage:
  python neural/teacher/lora.py --model <snapshot dir> --train <data>/pools/train.jsonl --dev <data>/pools/dev.jsonl --out <adapter dir> --loss lm --iters 3000
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_lm.models.qwen3_next as qwen3_next
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pool import expand, parts

SPAM = [
    "彩票",
    "时时彩",
    "快三",
    "11选5",
    "开奖",
    "棋牌",
    "博彩",
    "百家乐",
    "六合彩",
    "赌场",
    "娱乐城",
    "投注",
    "北京赛车",
    "pk10",
    "PK10",
]


def read_pools(path, limit, drop_spam):
    out, dropped = [], 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            if drop_spam and any(word in case["context"] or word in case["gold"] for word in SPAM):
                dropped += 1
                continue
            out.append(case)
            if limit and len(out) == limit:
                break
    return out, dropped


def encode(tokenizer, case, start):
    prefix = [start, *tokenizer.encode(case["context"], add_special_tokens=False)]
    return (
        prefix,
        [tokenizer.encode(text, add_special_tokens=False) for text in case["candidates"]],
        case["candidates"].index(case["gold"]),
    )


def save_adapter(model, path):
    mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))


def checkpoint(model, out, step, config):
    """A directory `mlx_lm.load(adapter_path=...)` can read for this step."""
    directory = out / f"step-{step:06d}"
    directory.mkdir(exist_ok=True)
    save_adapter(model, directory / "adapters.safetensors")
    shutil.copy(config, directory / "adapter_config.json")


class Log(TrainingCallback):
    def __init__(self, path, started):
        self.path = path
        self.started = started

    def write(self, row):
        row["wall_s"] = round(time.perf_counter() - self.started, 1)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), file=sys.stderr, flush=True)

    def on_train_loss_report(self, info):
        self.write({"kind": "train", **info})

    def on_val_loss_report(self, info):
        self.write({"kind": "val", **info})


GOLD_WIDTH = 24  # tokens; a gold is at most 16 characters


def gold_loss(model, batch, lengths):
    """Cross-entropy on the gold's tokens only. mlx-lm's `default_loss` also counts the padding token after the last one (its mask runs to `steps <= length`), so it is not used; and the output head runs only at the gold's positions, not over the whole sequence. The width is fixed so that mlx-lm's compiled step sees one shape per padded length."""
    _, body, head = parts(model)
    hidden = body(batch[:, :-1])
    offsets, ends = lengths[:, 0:1], lengths[:, 1:]
    steps = offsets - 1 + mx.arange(GOLD_WIDTH)[None, :]
    mask = steps < ends - 1
    steps = mx.minimum(steps, batch.shape[1] - 2)
    picked = mx.take_along_axis(hidden, steps[..., None], axis=1)
    targets = mx.take_along_axis(batch, steps + 1, axis=1)
    ce = nn.losses.cross_entropy(head(picked).astype(mx.float32), targets) * mask
    ntoks = mask.sum()
    return ce.sum() / ntoks, ntoks


def prefix_cache(model, text, body, prefix):
    """The cache after the prefix without its last token, from the current weights in eval mode and outside the gradient: the rank loss does not backpropagate into the context, only through the candidates' positions that attend to it."""
    if len(prefix) < 2:
        return None
    was_training = model.training
    model.eval()
    cache = text.make_cache()
    body(mx.array([prefix[:-1]]), cache=cache)
    arrays = []
    for layer in cache:
        if isinstance(layer, KVCache):
            layer.keys, layer.values = mx.stop_gradient(layer.keys), mx.stop_gradient(layer.values)
            arrays += [layer.keys, layer.values]
        else:
            layer.cache = [None if x is None else mx.stop_gradient(x) for x in layer.cache]
            arrays += [x for x in layer.cache if x is not None]
    mx.eval(arrays)
    if was_training:
        model.train()
    return cache


def attention_ops(queries, keys, values, cache, scale, mask, sinks=None):
    """Attention in plain ops for the rank loss's gradient: with fewer queries than keys mlx's fused kernel has no VJP. Same mask rules as the fused call ("causal" aligns the queries to the last keys)."""
    batch, query_heads, length, _ = queries.shape
    key_heads, size = keys.shape[1], keys.shape[2]
    q = (queries * scale).reshape(batch, key_heads, query_heads // key_heads, length, -1)
    scores = (q @ keys[:, :, None].swapaxes(-1, -2)).astype(mx.float32)
    if isinstance(mask, str):
        mask = (mx.arange(length)[:, None] + (size - length)) >= mx.arange(size)[None, :]
    if mask is not None:
        scores = mx.where(mask, scores, -1e9) if mask.dtype == mx.bool_ else scores + mask
    weights = mx.softmax(scores, axis=-1).astype(values.dtype)
    return (weights @ values[:, :, None]).reshape(batch, query_heads, length, -1)


def rank_loss(model, body, head, cache, last, candidates, gold, lm_weight):
    """Listwise cross-entropy over the pool's sums plus `lm_weight` times the gold's per-token LM loss. The prefix's last token leads every candidate row, so every candidate token, the first included, is scored with gradients."""
    n = len(candidates)
    # At least 4 positions: with fewer than the linear-attention layers' conv kernel (4), mlx's conv1d gradient comes out NaN. The padding is after every real token, so the causal layers do not let it reach them.
    width = max(4, *(len(tokens) for tokens in candidates))
    rows = mx.array([[last, *t] + [0] * (width - len(t)) for t in candidates])
    lengths = mx.array([len(t) for t in candidates])
    hidden = body(rows[:, :-1], cache=None if cache is None else expand(cache, n))
    logprobs = -nn.losses.cross_entropy(head(hidden).astype(mx.float32), rows[:, 1:])
    mask = mx.arange(width)[None, :] < lengths[:, None]
    sums = (logprobs * mask).sum(axis=1)
    listwise = mx.logsumexp(sums) - sums[gold]
    lm = -sums[gold] / lengths[gold]
    return listwise + lm_weight * lm, (listwise, lm)


def train_rank(model, tokenizer, start, pools, dev, args, out, config, log):
    text, body, head = parts(model)
    schedule = optim.cosine_decay(args.lr, args.iters, args.lr * 0.1)
    warm = optim.join_schedules([optim.linear_schedule(0.0, args.lr, args.warmup), schedule], [args.warmup])
    optimizer = optim.AdamW(learning_rate=warm, weight_decay=0.0)

    def loss_fn(model, cache, last, candidates, gold):
        return rank_loss(model, body, head, cache, last, candidates, gold, args.lm_weight)[0]

    value_and_grad = nn.value_and_grad(model, loss_fn)
    order = list(range(len(pools)))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    position = 0
    window = {"loss": 0.0, "tokens": 0, "pools": 0, "time": 0.0}
    model.train()
    for step in range(1, args.iters + 1):
        started = time.perf_counter()
        grads = None
        for _ in range(args.grad_accumulation):
            if position == len(order):
                rng.shuffle(order)
                position = 0
            prefix, candidates, gold = encode(tokenizer, pools[order[position]], start)
            position += 1
            prefix = prefix[-args.max_prefix :]
            cache = prefix_cache(model, text, body, prefix)
            fused, qwen3_next.scaled_dot_product_attention = qwen3_next.scaled_dot_product_attention, attention_ops
            loss, g = value_and_grad(model, cache, prefix[-1], candidates, gold)
            grads = g if grads is None else tree_map(lambda x, y: x + y, grads, g)
            mx.eval(loss, grads)
            qwen3_next.scaled_dot_product_attention = fused
            window["loss"] += loss.item()
            window["tokens"] += len(prefix) + sum(len(t) for t in candidates)
            window["pools"] += 1
        grads = tree_map(lambda x: x / args.grad_accumulation, grads)
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        window["time"] += time.perf_counter() - started
        if step % args.report_every == 0 or step == args.iters:
            log.write(
                {
                    "kind": "train",
                    "iteration": step,
                    "train_loss": window["loss"] / window["pools"],
                    "learning_rate": optimizer.learning_rate.item(),
                    "pools_per_second": window["pools"] / window["time"],
                    "tokens_per_second": window["tokens"] / window["time"],
                    "pools_seen": step * args.grad_accumulation,
                    "peak_memory": mx.get_peak_memory() / 1e9,
                }
            )
            window = {"loss": 0.0, "tokens": 0, "pools": 0, "time": 0.0}
        if step % args.eval_every == 0 or step == args.iters:
            model.eval()
            totals = [0.0, 0.0, 0]
            for case in dev:
                prefix, candidates, gold = encode(tokenizer, case, start)
                prefix = prefix[-args.max_prefix :]
                _, (listwise, lm) = rank_loss(
                    model,
                    body,
                    head,
                    prefix_cache(model, text, body, prefix),
                    prefix[-1],
                    candidates,
                    gold,
                    args.lm_weight,
                )
                mx.eval(listwise, lm)
                totals[0] += listwise.item()
                totals[1] += lm.item()
                totals[2] += 1
            log.write(
                {
                    "kind": "val",
                    "iteration": step,
                    "val_listwise": totals[0] / totals[2],
                    "val_lm_per_token": totals[1] / totals[2],
                }
            )
            model.train()
        if step % args.save_every == 0 or step == args.iters:
            checkpoint(model, out, step, config)
    save_adapter(model, out / "adapters.safetensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True, help="dev pools, for the validation loss only")
    parser.add_argument("--out", required=True)
    parser.add_argument("--loss", choices=["lm", "rank"], default="lm")
    parser.add_argument(
        "--examples", type=int, default=0, help="read only the first N training pools (after --drop-spam)"
    )
    parser.add_argument("--drop-spam", action="store_true")
    parser.add_argument("--iters", type=int, required=True, help="optimizer updates")
    parser.add_argument("--batch-size", type=int, default=8, help="lm: sequences per update")
    parser.add_argument("--grad-accumulation", type=int, default=8, help="rank: pools per update")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=20.0, help="mlx-lm's LoRA scale (it has no alpha)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=-1, help="LoRA on the last N blocks; -1 for all")
    parser.add_argument("--lm-weight", type=float, default=0.2, help="rank: weight of the gold's per-token LM loss")
    parser.add_argument(
        "--max-prefix",
        type=int,
        default=512,
        help="tokens of start + context kept (the stored context is at most 1024 bytes)",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--val-pools", type=int, default=200)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    started = time.perf_counter()
    # As in score_pool.py: with a different shape at every step mlx's buffer cache grows without bound (36 GB within minutes here); a bound keeps the reuse and frees the rest.
    mx.set_cache_limit(2 << 30)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(args.model)
    start = tokenizer.eos_token_id
    pools, dropped = read_pools(args.train, args.examples, args.drop_spam)
    dev, _ = read_pools(args.dev, args.val_pools, False)
    print(
        f"{len(pools)} training pools ({dropped} dropped as spam before the limit), {len(dev)} dev pools for the validation loss",
        file=sys.stderr,
        flush=True,
    )

    model.freeze()
    lora = {"rank": args.rank, "scale": args.scale, "dropout": args.dropout}
    layers = len(model.layers) if args.num_layers == -1 else args.num_layers
    linear_to_lora_layers(model, layers, lora)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    config = out / "adapter_config.json"
    config.write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": layers,
                "lora_parameters": lora,
                "model": args.model,
                "trainable_parameters": trainable,
                "training_pools": len(pools),
                "spam_dropped": dropped,
                **{key: value for key, value in vars(args).items() if key not in ("model",)},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    log = Log(out / "train-log.jsonl", started)
    log.write(
        {"kind": "start", "trainable_parameters": trainable, "training_pools": len(pools), "spam_dropped": dropped}
    )

    if args.loss == "rank":
        train_rank(model, tokenizer, start, pools, dev, args, out, config, log)
    else:

        def item(case):
            prefix = [start, *tokenizer.encode(case["context"], add_special_tokens=False)][-args.max_prefix :]
            return (prefix + tokenizer.encode(case["gold"], add_special_tokens=False), len(prefix))

        train_set = [item(case) for case in pools]
        val_set = [item(case) for case in dev]
        schedule = optim.cosine_decay(args.lr, args.iters, args.lr * 0.1)
        warm = optim.join_schedules([optim.linear_schedule(0.0, args.lr, args.warmup), schedule], [args.warmup])
        optimizer = optim.AdamW(learning_rate=warm, weight_decay=0.0)
        training = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=-1,
            steps_per_report=args.report_every,
            steps_per_eval=args.eval_every,
            steps_per_save=args.save_every,
            max_seq_length=args.max_prefix + 64,
            adapter_file=str(out / "adapters.safetensors"),
            grad_checkpoint=args.grad_checkpoint,
        )
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_set,
            val_dataset=val_set,
            args=training,
            loss=gold_loss,
            training_callback=log,
        )
        # mlx-lm writes `<step>_adapters.safetensors` next to the final file; give each its own loadable directory.
        for path in sorted(out.glob("*_adapters.safetensors")):
            step = int(path.name.split("_")[0])
            directory = out / f"step-{step:06d}"
            directory.mkdir(exist_ok=True)
            path.rename(directory / "adapters.safetensors")
            shutil.copy(config, directory / "adapter_config.json")
    log.write(
        {
            "kind": "end",
            "peak_memory": mx.get_peak_memory() / 1e9,
            "minutes": round((time.perf_counter() - started) / 60, 1),
        }
    )


if __name__ == "__main__":
    main()
