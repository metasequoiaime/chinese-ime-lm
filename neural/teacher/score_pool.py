"""Score candidate pools with a causal language model through `mlx-lm`, optionally with a LoRA adapter: the teacher's scorer.

The score is the one `eval/open-models/mlx_lm_pool.py` uses for the Qwen rows: the sum of the candidate's tokens' log-probabilities after the prefix, where the prefix is the start token (`--start`, default `eos`, the best start on the 52-case pool) followed by the case's context tokenized on its own (the pools keep at most the last 1024 bytes, MSIME's limit), and the candidate is tokenized on its own. The pick is the highest sum, the first one on a tie.

What differs is only how it is computed. `mlx_lm_pool.py` runs every candidate as its own row with the whole prefix in front, so a 300-token context goes through the model once per candidate. Here the prefix goes through once, its cache is copied to one row per candidate, and only the candidates' tokens go through after it; the first token of every candidate is scored from the prefix's last position. The model is causal, so the sums are the same up to floating-point rounding; `--check` scores the first cases both ways and reports the largest difference and any pick that changes.

`--no-context` drops the context (the start token stays), which is how the eval set's `pool-stripped.jsonl` scores the cases that have one.

The output is one JSON object per case (`id`, `pick`, `scores`, `ms`) and is appended to, so a run that stops can be started again with the same arguments and continues after the last case it wrote.

usage:
  python neural/teacher/score_pool.py --model <snapshot dir> [--adapter <dir>] --pool <pool.jsonl> --out <picks.jsonl>
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache


def start_token(tokenizer, kind):
    if kind == "none":
        return None
    if kind == "eos":
        return tokenizer.eos_token_id
    token = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if not isinstance(token, int) or token == tokenizer.unk_token_id:
        raise SystemExit("this tokenizer has no <|endoftext|>")
    return token


def parts(model):
    """The text model (for its cache), the transformer body and the output head, as in `mlx_lm_pool.py`."""
    text = getattr(model, "language_model", model)
    head = text.lm_head if hasattr(text, "lm_head") else text.model.embed_tokens.as_linear
    return text, text.model, head


def log_softmax_at(head, hidden, targets):
    logits = head(hidden).astype(mx.float32)
    return mx.take_along_axis(logits, targets[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)


def score_full(body, head, prefix, candidates):
    """`mlx_lm_pool.py`'s computation, kept as the reference for `--check`."""
    width = max(len(prefix) + len(tokens) for tokens in candidates)
    rows, positions, targets, owners = [], [], [], []
    for row, tokens in enumerate(candidates):
        sequence = [*prefix, *tokens]
        rows.append(sequence + [0] * (width - len(sequence)))
        first = 0 if prefix else 1
        for k in range(first, len(tokens)):
            positions.append(row * width + len(prefix) + k - 1)
            targets.append(tokens[k])
            owners.append(row)
    if not positions:
        return [0.0] * len(candidates)
    hidden = body(mx.array(rows))
    hidden = hidden.reshape(-1, hidden.shape[-1])[mx.array(positions)]
    logprobs = log_softmax_at(head, hidden, mx.array(targets))
    totals = mx.zeros((len(candidates),), dtype=mx.float32).at[mx.array(owners)].add(logprobs)
    mx.eval(totals)
    return totals.tolist()


def expand(cache, n):
    """One copy of a batch-1 cache per candidate."""
    out = []
    for layer in cache:
        if isinstance(layer, KVCache):
            copy = KVCache()
            copy.keys = mx.repeat(layer.keys[..., : layer.offset, :], n, axis=0)
            copy.values = mx.repeat(layer.values[..., : layer.offset, :], n, axis=0)
            copy.offset = layer.offset
        elif isinstance(layer, ArraysCache):
            copy = ArraysCache(len(layer.cache))
            copy.cache = [None if a is None else mx.repeat(a, n, axis=0) for a in layer.cache]
        else:
            raise SystemExit(f"no cache copy for {type(layer).__name__}")
        out.append(copy)
    return out


def score_cached(text, body, head, prefix, candidates):
    """The same sums with the prefix run once."""
    if not prefix:
        return score_full(body, head, prefix, candidates)
    cache = text.make_cache()
    last = body(mx.array([prefix]), cache=cache)[0, -1:]
    firsts = log_softmax_at(
        head, mx.repeat(last, len(candidates), axis=0), mx.array([tokens[0] for tokens in candidates])
    )
    width = max(len(tokens) for tokens in candidates)
    if width == 1:
        mx.eval(firsts)
        return firsts.tolist()
    rows, positions, targets, owners = [], [], [], []
    for row, tokens in enumerate(candidates):
        rows.append(tokens + [0] * (width - len(tokens)))
        for k in range(1, len(tokens)):
            positions.append(row * width + k - 1)
            targets.append(tokens[k])
            owners.append(row)
    hidden = body(mx.array(rows), cache=expand(cache, len(candidates)))
    hidden = hidden.reshape(-1, hidden.shape[-1])[mx.array(positions)]
    logprobs = log_softmax_at(head, hidden, mx.array(targets))
    totals = firsts.at[mx.array(owners)].add(logprobs)
    mx.eval(totals)
    return totals.tolist()


def pick_of(scores):
    return max(range(len(scores)), key=lambda k: (scores[k], -k))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", required=True, help="a local snapshot directory or a Hugging Face repository mlx-lm can load"
    )
    parser.add_argument("--adapter", help="a LoRA adapter directory written by neural/teacher/lora.py")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", choices=["none", "endoftext", "eos"], default="eos")
    parser.add_argument("--no-context", action="store_true", help="score every case without its context")
    parser.add_argument(
        "--only-context",
        action="store_true",
        help="only the cases that have a context (with --no-context: the eval set's stripped pool)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only the first N cases of the pool (the pools are shuffled, so a prefix is a random sample)",
    )
    parser.add_argument(
        "--check", type=int, default=0, help="score the first N cases both ways, report the differences, and stop"
    )
    args = parser.parse_args()

    mx.set_cache_limit(2 << 30)
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    model.eval()
    text, body, head = parts(model)
    start = start_token(tokenizer, args.start)
    with open(args.pool, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if args.only_context:
        cases = [case for case in cases if case.get("context")]
    if args.limit:
        cases = cases[: args.limit]

    def encode(case):
        context = "" if args.no_context else case.get("context", "")
        prefix = [*([] if start is None else [start]), *tokenizer.encode(context, add_special_tokens=False)]
        return prefix, [tokenizer.encode(text, add_special_tokens=False) for text in case["candidates"]]

    if args.check:
        worst, changed = 0.0, []
        for case in cases[: args.check]:
            prefix, candidates = encode(case)
            a = score_full(body, head, prefix, candidates)
            b = score_cached(text, body, head, prefix, candidates)
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b, strict=True)))
            if pick_of(a) != pick_of(b):
                gap = sorted(a, reverse=True)
                changed.append({"id": case["id"], "reference_gap": gap[0] - gap[1]})
        print(json.dumps({"cases": min(args.check, len(cases)), "max_abs_diff": worst, "pick_changed": changed}))
        return

    out = Path(args.out)
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            done = {json.loads(line)["id"] for line in handle if line.strip()}
    todo = [case for case in cases if case["id"] not in done]
    print(f"{len(cases)} cases, {len(done)} already scored, {len(todo)} to go", file=sys.stderr, flush=True)

    started_all = time.perf_counter()
    times = []
    with out.open("a", encoding="utf-8") as handle:
        for i, case in enumerate(todo, 1):
            prefix, candidates = encode(case)
            started = time.perf_counter()
            scores = score_cached(text, body, head, prefix, candidates)
            ms = (time.perf_counter() - started) * 1000
            times.append(ms)
            handle.write(
                json.dumps({"id": case["id"], "pick": pick_of(scores), "scores": scores, "ms": round(ms, 2)}) + "\n"
            )
            if i % 200 == 0 or i == len(todo):
                handle.flush()
                elapsed = time.perf_counter() - started_all
                print(
                    f"{i}/{len(todo)} in {elapsed:.0f} s, p50 {statistics.median(times):.1f} ms, peak {mx.get_peak_memory() / 1e9:.1f} GB",
                    file=sys.stderr,
                    flush=True,
                )

    with out.open(encoding="utf-8") as handle:
        rows = {row["id"]: row for row in map(json.loads, handle) if row}
    hits = sum(case["candidates"][rows[case["id"]]["pick"]] == case["gold"] for case in cases if case["id"] in rows)
    print(
        f"{args.model} adapter {args.adapter} (start {args.start}{', no context' if args.no_context else ''}): {hits}/{len(cases)} hits",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
