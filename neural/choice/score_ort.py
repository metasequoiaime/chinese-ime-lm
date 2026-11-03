"""Pick on a pool file with an exported `lm_step.ort` itself, the file a release ships, and report the hits with and without a context.

`evaluate.py --lm-only` exports its own two graphs and quantizes them to int8, so its numbers are not those of the released uint8 single graph (#100). This scores the release directory as published: each case from an empty cache, the context cut the way MSIME hands it to the reranker (`rerank_context` in MSIME's `crates/engine/src/ordering.rs`: a window of `context_chars + 16` characters less the longest candidate and one, kept in steps of 16), then its last `context_chars` characters encoded after BOS, and all candidates scored in one keystroke. The pick is the highest mean log-probability per character, the first on a tie.

Writes one line per case, `{id, pick, scores}`, like `evaluate.py` and `reference/examples/pool.rs`, and prints the hits.

usage:
  python neural/choice/score_ort.py --model <dir with lm_step.ort, model.json, vocab.json> --pool <pool.jsonl> --out <picks.jsonl> [--threads 1]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_step import NAMES, context_feed, keystroke_feed  # noqa: E402

CONTEXT_STEP = 16
CANDIDATE_ROOM = 16


def rerank_context(context, window, longest):
    """MSIME's `rerank_context`: the tail of the committed text a reranker with a `window`-character context reads beside a `longest`-character candidate."""
    room = max(0, window - (longest + 1))
    keep = room // CONTEXT_STEP * CONTEXT_STEP
    if len(context) <= room:
        return context
    return context[len(context) - keep :]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    import onnxruntime as ort

    model = Path(args.model)
    info = json.loads((model / "model.json").read_text(encoding="utf-8"))
    tokens = json.loads((model / "vocab.json").read_text(encoding="utf-8"))["tokens"]
    ids = {t: i for i, t in enumerate(tokens) if len(t) == 1}
    chars, bos, unk = info["context_chars"], info["bos"], info["unk"]

    def encode(text):
        return [ids.get(c, unk) for c in text]

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(model / "lm_step.ort"), options)
    empty = {
        "cache_k": np.zeros((info["layers"], info["heads"], 0, info["head_dim"]), np.float32),
        "cache_v": np.zeros((info["layers"], info["heads"], 0, info["head_dim"]), np.float32),
        "cache_h": np.zeros((0, info["width"]), np.float32),
        "cache_z": np.zeros((0,), np.float32),
    }
    hits = {True: [0, 0], False: [0, 0]}
    with open(args.pool, encoding="utf-8") as handle, open(args.out, "w", encoding="utf-8") as out:
        for line in handle:
            case = json.loads(line)
            longest = max(len(c) for c in case["candidates"])
            text = rerank_context(case.get("context", ""), chars + CANDIDATE_ROOM, longest)[-chars:]
            context = [bos, *encode(text)]
            result = session.run(None, empty | context_feed(context))
            cache = dict(zip(NAMES[:4], result[1:], strict=True))
            feed, _ = keystroke_feed(len(context), [encode(c) for c in case["candidates"]], [])
            scores = [float(s) for s in session.run(None, cache | feed)[0]]
            pick = max(range(len(scores)), key=lambda i: (scores[i], -i))
            out.write(json.dumps({"id": case["id"], "pick": pick, "scores": scores}) + "\n")
            tally = hits[bool(case.get("context"))]
            tally[0] += case["candidates"][pick] == case["gold"]
            tally[1] += 1
    for has, (h, n) in hits.items():
        if n:
            print(f"{'context' if has else 'none'}: {h}/{n} = {100 * h / n:.1f}%")


if __name__ == "__main__":
    main()
