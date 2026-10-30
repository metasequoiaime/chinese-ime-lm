"""Score the candidate pool with a general-purpose causal language model through `mlx-lm`.

This is the scorer for the Qwen rows. Each candidate is scored as the sum of its tokens'
log-probabilities after the case's context; candidates in a pool have the same length, so the sum
ranks them as a per-character mean would. The pick is the highest sum, the first one on a tie.

What comes before the candidate decides how its first token is scored, and the result depends on it
(`docs/measurements.md` has the numbers), so it is an option rather than a hidden choice:

- `--start none`: nothing but the context. With no context the first token has no distribution to
  be scored from and is left out of the sum; this is how the 2026-09-19 measurement ran, and it is
  the default so that the published rows reproduce. With a context every token is scored.
- `--start endoftext`: `<|endoftext|>` before the context, the separator between pretraining
  documents in Qwen's stream.
- `--start eos`: the tokenizer's `eos_token` before the context, which in the instruct and hybrid
  Qwen releases is the chat turn's `<|im_end|>`.

A general model has no input method of its own, so the context is the case's whole context (the set
keeps at most the last 1024 bytes, MSIME's limit), tokenized on its own and followed by the
candidate tokenized on its own, so that every candidate in a pool is scored after the same tokens.

The candidates of a case go through the model as one batch, padded on the right; the model is
causal, so the padding cannot reach a scored position. The time is that one batched forward pass
per case, the median over passes: the cost of a decision with no state kept between decisions and
no reuse of the context across keystrokes. (The 2026-09-19 measurement ran one forward pass per
candidate and summed them; on the 52-case pool, with no context, the picks are the same.)

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


def parts(model):
    """The transformer body and the output head, so that only the scored positions go through the head."""
    text = getattr(model, "language_model", model)
    head = text.lm_head if hasattr(text, "lm_head") else text.model.embed_tokens.as_linear
    return text.model, head


def score(body, head, prefix, candidates):
    """Sum of log P(token | everything before it) over each candidate's tokens, after `prefix`.

    With an empty prefix nothing precedes a candidate and its first token goes unscored.
    """
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
        # No prefix and every candidate is a single token: nothing is scored, every sum is 0, and the tie goes to the first.
        return [0.0] * len(candidates)
    hidden = body(mx.array(rows))
    hidden = hidden.reshape(-1, hidden.shape[-1])[mx.array(positions)]
    logits = head(hidden).astype(mx.float32)
    logprobs = mx.take_along_axis(logits, mx.array(targets)[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)
    totals = mx.zeros((len(candidates),), dtype=mx.float32).at[mx.array(owners)].add(logprobs)
    mx.eval(totals)
    return totals.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a Hugging Face repository mlx-lm can load")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--start", choices=["none", "endoftext", "eos"], default="none")
    args = parser.parse_args()

    # mlx keeps freed buffers for reuse, and with a different shape in every case the cache grows without bound (over 30 GB within the first 100 cases of eval/sentences-ctx-v1, enough for macOS to kill the process). A bound keeps the reuse and frees the rest; the scores do not change.
    mx.set_cache_limit(2 << 30)
    model, tokenizer = load(args.model)
    body, head = parts(model)
    start = start_token(tokenizer, args.start)
    with open(args.pool, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    prefixes = [
        [*([] if start is None else [start]), *tokenizer.encode(case.get("context", ""), add_special_tokens=False)] for case in cases
    ]
    encoded = [[tokenizer.encode(text, add_special_tokens=False) for text in case["candidates"]] for case in cases]

    for prefix, case in zip(prefixes[:8], encoded[:8]):  # untimed: weights paged in, kernels compiled
        score(body, head, prefix, case)
    samples = [[] for _ in cases]
    scores = [None] * len(cases)
    for _ in range(max(1, args.passes)):
        for i, (prefix, case) in enumerate(zip(prefixes, encoded, strict=True)):
            started = time.perf_counter()
            scores[i] = score(body, head, prefix, case)
            samples[i].append((time.perf_counter() - started) * 1000)

    hits = 0
    decisions = []
    for case, scored, times in zip(cases, scores, samples, strict=True):
        pick = max(range(len(scored)), key=lambda k: (scored[k], -k))
        hits += case["candidates"][pick] == case["gold"]
        ms = statistics.median(times)
        decisions.append(ms)
        print(json.dumps({"id": case["id"], "pick": pick, "scores": scored, "ms": ms}))
    print(
        f"{args.model} (start {args.start}): {hits}/{len(cases)} hits; decision p50 {statistics.median(decisions):.2f} ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
