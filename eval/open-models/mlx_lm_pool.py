"""Score the candidate pool with a general-purpose causal language model through `mlx-lm`.

This is the scorer for the Qwen rows. Each candidate is scored on its own, by one forward pass, as the
sum of its tokens' log-probabilities; candidates in a pool have the same length, so the sum ranks them
as a per-character mean would. The pick is the highest sum, the first one on a tie.

What comes before the candidate decides how its first token is scored, and the result depends on it
(`docs/measurements.md` has the numbers), so it is an option rather than a hidden choice:

- `--start none`: nothing. The first token has no distribution to be scored from and is left out of
  the sum. This is how the 2026-09-19 measurement ran, and it is the default so that the published
  rows reproduce.
- `--start endoftext`: `<|endoftext|>`, the separator between pretraining documents in Qwen's stream.
- `--start eos`: the tokenizer's `eos_token`, which in the instruct and hybrid Qwen releases is the
  chat turn's `<|im_end|>`.

The time is per candidate: one forward pass, with no batching and no reuse of a prefix the candidates
share, and the figure is the median over passes. A decision costs one such pass per candidate.

usage:
  python eval/open-models/mlx_lm_pool.py --model mlx-community/Qwen3-0.6B-8bit --pool eval/open-models/pool.jsonl > picks.jsonl
"""

import argparse
import json
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load


def start_token(tokenizer, kind):
    if kind == "none":
        return None
    if kind == "eos":
        return tokenizer.eos_token_id
    token = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if not isinstance(token, int) or token == tokenizer.unk_token_id:
        raise SystemExit("this tokenizer has no <|endoftext|>")
    return token


def score(model, start, tokens):
    """Sum of log P(token | the tokens before it) over the candidate's tokens.

    With `start` None nothing precedes the candidate and its first token goes unscored.
    """
    ids = mx.array([[*([] if start is None else [start]), *tokens]])
    logits = model(ids)[0, :-1, :].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = tokens if start is not None else tokens[1:]
    picked = mx.take_along_axis(logprobs, mx.array(targets)[:, None], axis=-1)
    total = mx.sum(picked)
    mx.eval(total)
    return total.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a Hugging Face repository mlx-lm can load")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--start", choices=["none", "endoftext", "eos"], default="none")
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    start = start_token(tokenizer, args.start)
    with open(args.pool, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    encoded = [[tokenizer.encode(text, add_special_tokens=False) for text in case["candidates"]] for case in cases]

    for tokens in encoded[0]:  # untimed: weights paged in, kernels compiled
        score(model, start, tokens)
    samples = [[[] for _ in case] for case in encoded]
    scores = [[0.0] * len(case) for case in encoded]
    for _ in range(max(1, args.passes)):
        for i, case in enumerate(encoded):
            for j, tokens in enumerate(case):
                started = time.perf_counter()
                scores[i][j] = score(model, start, tokens)
                samples[i][j].append((time.perf_counter() - started) * 1000)

    hits = 0
    per_candidate = []
    decisions = []
    for case, scored, times in zip(cases, scores, samples, strict=True):
        pick = max(range(len(scored)), key=lambda k: (scored[k], -k))
        hits += case["candidates"][pick] == case["gold"]
        medians = [statistics.median(t) for t in times]
        per_candidate.extend(medians)
        decisions.append(sum(medians))
        print(json.dumps({"id": case["id"], "pick": pick, "scores": scored, "ms": sum(medians)}))
    print(
        f"{args.model} (start {args.start}): {hits}/{len(cases)} hits; per candidate p50 {statistics.median(per_candidate):.2f} ms; decision p50 {statistics.median(decisions):.2f} ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
