"""P3 of `docs/choice-model.md`: export a trained choice model to int8 ONNX and pick on a pool file with ONNX Runtime, the way the input method will run it.

Writes one line per case, `{id, pick, scores, ms}` like `reference/examples/pool.rs`, so `neural/teacher/compare.py` takes it as a picker. The context graph runs once per case (at inference it runs when text is committed) and the keystroke graph scores the page; `ms` is the keystroke graph alone. The pick is the highest score, the first on a tie.

usage:
  python neural/choice/evaluate.py --run <pools-phase dir> --pool <pools/ctx-v1.jsonl> --out <picks.jsonl> [--threads 2] [--strip-context]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latency import export, session  # noqa: E402
from model import END, ChoiceConfig, ChoiceModel, letter_ids  # noqa: E402
from train import CONTEXT_CHARS, LETTERS_MAX, Encoder  # noqa: E402


def load(run):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = ChoiceModel(ChoiceConfig(**ckpt["config"])).eval()
    model.load_state_dict(ckpt["model"])
    vocab = json.loads((Path(run) / "vocab.json").read_text(encoding="utf-8"))["tokens"]
    return model, vocab, ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--strip-context", action="store_true", help="score only the cases that have a context, without it (the eval set's context-stripped subset)")
    args = parser.parse_args()

    run = Path(args.run)
    model, vocab, ckpt = load(run)
    onnx_dir = run / "onnx"
    if not (onnx_dir / "keystroke.int8.onnx").exists():
        onnx_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            export(model, onnx_dir)
    ctx, key = session(onnx_dir / "context.int8.onnx", args.threads), session(onnx_dir / "keystroke.int8.onnx", args.threads)
    encoder = Encoder(vocab)

    empty = None
    hits = n = 0
    with open(args.pool, encoding="utf-8") as handle, open(args.out, "w", encoding="utf-8") as out:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            context_text = case.get("context", "")
            if args.strip_context:
                if not context_text:
                    continue
                context_text = ""
            context = np.array(encoder.chars(context_text[-CONTEXT_CHARS:]), dtype=np.int64)
            if len(context):
                k, v = ctx.run(None, {"context": context})
            else:
                if empty is None:
                    k0, v0 = ctx.run(None, {"context": np.array([END], dtype=np.int64)})
                    empty = (k0[:, :, :0], v0[:, :, :0])
                k, v = empty
            letters = np.array(letter_ids(case["input"])[:LETTERS_MAX], dtype=np.int64)
            bodies = [encoder.chars(c) + [END] for c in case["candidates"]]
            width = max(len(b) for b in bodies)
            candidates = np.zeros((len(bodies), width), dtype=np.int64)
            for i, b in enumerate(bodies):
                candidates[i, : len(b)] = b
            ends = np.array([len(b) - 1 for b in bodies], dtype=np.int64)
            started = time.perf_counter()
            scores = key.run(None, {"past_k": k, "past_v": v, "letters": letters, "candidates": candidates, "ends": ends})[0]
            ms = (time.perf_counter() - started) * 1000
            pick = int(np.argmax(scores))
            n += 1
            hits += case["candidates"][pick] == case["gold"]
            out.write(json.dumps({"id": case["id"], "pick": pick, "scores": [float(s) for s in scores], "ms": round(ms, 3)}) + "\n")
    print(f"{hits}/{n} hits ({100 * hits / max(1, n):.1f}%), checkpoint step {ckpt.get('step')}, dev hits {ckpt.get('dev_hits')}", file=sys.stderr)


if __name__ == "__main__":
    main()
