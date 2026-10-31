"""Fine-tune a student run from `neural/train.py` on candidate pools, with the teacher's scores (distillation), the gold, or both.

The student is used through one number: the mean log-probability per character of a candidate after BOS and the tail of the committed text (`reference/src/lib.rs`, `SentenceModel::score`), the highest one picked. So it is trained on that number. For one pool with candidates c_j, the student's score s_j is the summed log-probability of c_j's characters, computed on the tokens the runtime would feed (BOS, the context cut by MSIME's `rerank_context` and then by `prefix_tokens`, the candidate); the candidates of a pool have equal length, so the sum orders them as the mean does. Per pool:

- `kd`: T^2 KL(softmax(t / T) || softmax(s / T)), where t_j is the teacher's score for c_j (`score_pool.py`'s summed log-probabilities). The teacher's whole ranking is the target, including how close the wrong candidates were.
- `gold`: -log softmax(s)[gold], the listwise loss with the gold alone.

The loss is `--kd-weight` * kd + `--gold-weight` * gold, averaged over pools, plus `--lm-weight` times the language-model loss on a batch of corpus windows (the run's own `tokens.u16`, sampled as `train.py --windows document` does), so the model stays a language model on text outside the pools.

Every `--eval-every` steps the student picks on the validation pools the way `reference/examples/pool.rs` does (highest mean log-probability per character, the first on a tie) and the checkpoint with the most hits is kept as `checkpoint.pt` in `train.py`'s format, so `neural/export.py` exports it unchanged. `config.json` and `vocab.json` are copied from the starting run, with the distillation settings added.

usage:
  python neural/teacher/distill.py --run <student run> --pools <data>/pools/kd.jsonl --teacher <data>/teacher/kd/teacher.jsonl \\
    --val <data>/teacher/dev-select.jsonl --out <student distilled run> --kd-weight 1 --gold-weight 0.5
"""

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import BOS, PAD, UNK, CharLM, Config  # noqa: E402
from train import context_batches, learning_rate, pick_device  # noqa: E402

# MSIME's `RERANK_CONTEXT_STEP` (crates/engine/src/ordering.rs).
RERANK_CONTEXT_STEP = 16


def rerank_context(context, window, longest):
    """MSIME's `rerank_context`: the context whole if it fits, else its tail cut in steps of 16 characters."""
    room = max(0, window - (longest + 1))
    keep = room // RERANK_CONTEXT_STEP * RERANK_CONTEXT_STEP
    if len(context) <= room:
        return context
    return context[len(context) - keep :] if keep else ""


class Encoder:
    def __init__(self, tokens, window):
        self.index = {ch: i for i, ch in enumerate(tokens)}
        self.window = window

    def encode(self, text):
        return [self.index.get(ch, UNK) for ch in text]

    def rows(self, case):
        """The token rows the runtime scores for one pool, and where each candidate's characters start."""
        candidates = case["candidates"]
        longest = max(len(c) for c in candidates)
        context = rerank_context(case.get("context", ""), self.window, longest)
        room = max(0, self.window - (longest + 1))
        encoded = self.encode(context)
        prefix = [BOS] + encoded[len(encoded) - room :] if room else [BOS]
        limit = max(1, self.window - len(prefix))
        return [prefix + self.encode(c)[:limit] for c in candidates], len(prefix)


def candidate_scores(model, encoder, cases, device):
    """Summed log-probability of every candidate of every case, as one flat tensor, and each case's slice of it."""
    rows, starts, spans = [], [], []
    for case in cases:
        made, start = encoder.rows(case)
        spans.append((len(rows), len(rows) + len(made)))
        rows.extend(made)
        starts.extend([start] * len(made))
    # Widths rounded up to a multiple of 32: on MPS each new shape keeps its own cached buffers, and a run of exact widths grew the cache until it ran out of memory.
    width = min(encoder.window, -(-max(len(r) for r in rows) // 32) * 32)
    x = torch.full((len(rows), width - 1), PAD, dtype=torch.long)
    y = torch.full((len(rows), width - 1), PAD, dtype=torch.long)
    mask = torch.zeros((len(rows), width - 1), dtype=torch.bool)
    for i, (row, start) in enumerate(zip(rows, starts)):
        t = torch.tensor(row)
        x[i, : len(row) - 1] = t[:-1]
        y[i, : len(row) - 1] = t[1:]
        # Position p predicts token p + 1; the candidate's tokens are row[start:].
        mask[i, start - 1 : len(row) - 1] = True
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    # The head only at the candidates' positions: a pool's rows are mostly context, and the full logits (rows x positions x vocab) of a batch run to gigabytes.
    picked = model.hidden(x)[mask]
    logp = torch.log_softmax(model.head(picked).float(), dim=-1).gather(-1, y[mask].unsqueeze(-1)).squeeze(-1)
    row = torch.arange(len(rows), device=device).unsqueeze(1).expand_as(mask)[mask]
    sums = torch.zeros(len(rows), device=device).index_add(0, row, logp)
    lengths = mask.sum(dim=1)
    return sums, lengths, spans


def pool_loss(sums, spans, cases, teacher, args):
    kd_terms, gold_terms = [], []
    for case, (a, b) in zip(cases, spans):
        s = sums[a:b]
        if args.kd_weight:
            t = torch.tensor(teacher[case["id"]], dtype=torch.float32, device=s.device)
            temp = args.temperature
            kd_terms.append(temp * temp * F.kl_div(F.log_softmax(s / temp, dim=0), F.log_softmax(t / temp, dim=0), log_target=True, reduction="sum"))
        if args.gold_weight:
            gold = case["candidates"].index(case["gold"])
            gold_terms.append(-F.log_softmax(s, dim=0)[gold])
    zero = sums.new_zeros(())
    kd = torch.stack(kd_terms).mean() if kd_terms else zero
    gold = torch.stack(gold_terms).mean() if gold_terms else zero
    return kd, gold


@torch.no_grad()
def evaluate(model, encoder, cases, device, batch):
    """Hits picking like `pool.rs`: highest mean log-probability per character, the first on a tie."""
    model.eval()
    hits = 0
    for i in range(0, len(cases), batch):
        chunk = cases[i : i + batch]
        sums, lengths, spans = candidate_scores(model, encoder, chunk, device)
        means = (sums / lengths.clamp(min=1)).cpu().tolist()
        for case, (a, b) in zip(chunk, spans):
            row = means[a:b]
            pick = row.index(max(row))
            hits += case["candidates"][pick] == case["gold"]
    model.train()
    return hits


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="the student run to start from (train.py's output directory)")
    parser.add_argument("--pools", required=True)
    parser.add_argument("--teacher", help="score_pool.py's output on --pools; required when --kd-weight > 0")
    parser.add_argument("--val", required=True, help="validation pools (dev-select)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--gold-weight", type=float, default=0.0)
    parser.add_argument("--lm-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-pools", type=int, default=0, help="use only the first N pools (0: all; with --teacher, all it scored)")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=32, help="pools per step")
    parser.add_argument("--lm-batch", type=int, default=32, help="corpus windows per step for the LM loss")
    # The pretraining ends at a tenth of its 3e-4 peak; 1e-4 shook the pretrained model (dev hits 1300 -> 1260 and LM loss 3.85 -> 4.05 by step 500), so the fine-tuning starts from where it left off.
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--progress", help="the dashboard's progress file")
    parser.add_argument("--progress-key", default="distill")
    parser.add_argument("--progress-label", default="", help="shown before the step count, to tell runs that share a stage apart")
    args = parser.parse_args()
    if args.kd_weight and not args.teacher:
        parser.error("--kd-weight needs --teacher")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    cfg = Config(**start["config"])
    tokens = json.loads((run / "vocab.json").read_text(encoding="utf-8"))["tokens"]
    encoder = Encoder(tokens, cfg.context)
    device = pick_device()
    model = CharLM(cfg).to(device)
    model.load_state_dict(start["model"])

    pools = read_jsonl(args.pools)
    teacher = {}
    if args.teacher:
        for row in read_jsonl(args.teacher):
            teacher[row["id"]] = row["scores"]
        pools = [p for p in pools if p["id"] in teacher]
    if args.max_pools:
        pools = pools[: args.max_pools]
    val = read_jsonl(args.val)
    steps = math.ceil(len(pools) * args.epochs / args.batch)
    print(f"{len(pools)} pools, {len(val)} validation pools, {steps} steps of {args.batch} on {device}", flush=True)

    base_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    base_config["distill"] = {k: v for k, v in vars(args).items() if not k.startswith("progress")}
    base_config["distill"]["pools_used"] = len(pools)
    base_config["distill"]["start_step"] = start["step"]
    (out / "config.json").write_text(json.dumps(base_config, ensure_ascii=False), encoding="utf-8")
    shutil.copy(run / "vocab.json", out / "vocab.json")

    lm_tokens = np.memmap(run / "tokens.u16", dtype=np.uint16, mode="r")
    split = int(len(lm_tokens) * 0.999)
    lm_stream = context_batches(lm_tokens[:split], args.lm_batch, cfg.context, device, torch.Generator().manual_seed(args.seed))
    lm_val = context_batches(lm_tokens[split:], 64, cfg.context, device, torch.Generator().manual_seed(7))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    log = open(out / "distill-log.jsonl", "a", encoding="utf-8")

    def record(row):
        log.write(json.dumps(row) + "\n")
        log.flush()

    if args.progress:
        note_progress(args.progress, args.progress_key, f"{args.progress_label}准备中：先测起点")
    best = evaluate(model, encoder, val, device, 64)
    print(f"start: {best}/{len(val)} validation hits", flush=True)
    record({"kind": "val", "step": 0, "hits": best, "n": len(val), "t": time.time()})
    torch.save({**start, "step": start["step"], "dev_hits": best}, out / "checkpoint.pt")
    order = list(range(len(pools)))
    random.shuffle(order)
    cursor = 0
    started = time.time()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, args.lr, args.warmup)
        if cursor + args.batch > len(order):
            random.shuffle(order)
            cursor = 0
        cases = [pools[i] for i in order[cursor : cursor + args.batch]]
        cursor += args.batch
        sums, _, spans = candidate_scores(model, encoder, cases, device)
        kd, gold = pool_loss(sums, spans, cases, teacher, args)
        loss = args.kd_weight * kd + args.gold_weight * gold
        lm = None
        if args.lm_weight:
            x, y = next(lm_stream)
            _, lm = model(x, y)
            loss = loss + args.lm_weight * lm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "mps" and (step + 1) % 50 == 0:
            torch.mps.empty_cache()
        if (step + 1) % 50 == 0:
            elapsed = time.time() - started
            record({"kind": "train", "step": step + 1, "steps": steps, "loss": loss.item(), "kd": kd.item(), "gold": gold.item(), "lm": lm.item() if lm is not None else None, "wall_s": elapsed, "t": time.time()})
            if args.progress:
                update_progress(args.progress, args.progress_key, step + 1, steps, elapsed, args.progress_label)
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            hits = evaluate(model, encoder, val, device, 64)
            model.eval()
            with torch.no_grad():
                lm_loss = sum(model(*next(lm_val))[1].item() for _ in range(10)) / 10
            model.train()
            print(f"step {step + 1}/{steps}  loss {loss.item():.4f}  val {hits}/{len(val)}  lm {lm_loss:.4f}  {(time.time() - started) / 60:.1f} min", flush=True)
            record({"kind": "val", "step": step + 1, "hits": hits, "n": len(val), "lm_val": lm_loss, "t": time.time()})
            if hits > best:
                best = hits
                torch.save({"config": dataclasses.asdict(cfg), "model": model.state_dict(), "val": lm_loss, "step": start["step"] + step + 1, "dev_hits": hits}, out / "checkpoint.pt")
    print(f"best {best}/{len(val)} validation hits", flush=True)
    record({"kind": "end", "best": best, "n": len(val), "minutes": round((time.time() - started) / 60, 1)})
    if args.progress:
        note_progress(args.progress, args.progress_key, f"{args.progress_label}跑完，验证集最好 {best} / {len(val)}")


def note_progress(path, key, detail):
    """A running stage between step reports (the start evaluation, the gap before the next run), so the dashboard does not keep the last run's step count."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    data[key] = {**data.get(key, {}), "status": "running", "detail": detail, "eta_s": None, "live": None}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def update_progress(path, key, step, steps, elapsed, label=""):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    per = elapsed / step
    entry = data.get(key, {})
    entry.update({"status": "running", "progress": step / steps, "detail": f"{label}{step} / {steps} 步", "eta_s": (steps - step) * per, "live": {"last_it": step, "total": steps, "sec_per_it": per, "log_at": time.time()}})
    data[key] = entry
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
