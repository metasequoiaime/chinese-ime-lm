"""P3 of `docs/choice-model.md`: export a trained choice model to int8 ONNX and pick on a pool file with ONNX Runtime, the way the input method will run it.

With `--lm-only` the decoder is scored as a character LM instead (`ChoiceModel.lm_context` / `lm_score`, the way `reference/examples/pool.rs` scores the 2.0 models): the context graph reads BOS and the tail of the committed text, the keystroke graph the candidates alone, and a pick is the highest mean log-probability per character. `--run` can then be an LM-phase run.

Writes one line per case, `{id, pick, scores, ms}` like `reference/examples/pool.rs`, so `neural/teacher/compare.py` takes it as a picker. The context graph runs once per case (at inference it runs when text is committed) and the keystroke graph scores the page; `ms` is the keystroke graph alone. The pick is the highest score, the first on a tie.

usage:
  python neural/choice/evaluate.py --run <pools-phase dir> --pool <pools/ctx-v1.jsonl> --out <picks.jsonl> [--threads 2] [--strip-context] [--lm-only]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latency import export, session  # noqa: E402
from model import END, ChoiceConfig, ChoiceModel, LmContextGraph, LmKeystrokeGraph, letter_ids  # noqa: E402
from train import CONTEXT_CHARS, LETTERS_MAX, Encoder  # noqa: E402


def load(run):
    ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = ChoiceModel(ChoiceConfig(**ckpt["config"])).eval()
    model.load_state_dict(ckpt["model"])
    vocab = json.loads((Path(run) / "vocab.json").read_text(encoding="utf-8"))["tokens"]
    return model, vocab, ckpt


def export_lm(model, out):
    context = torch.tensor([END, 3, 4, 5])
    past_k, past_v, first = model.lm_context(context)
    candidates = torch.tensor([[3, 4, 5], [6, 7, 0]])
    lengths = torch.tensor([3, 2])
    torch.onnx.export(LmContextGraph(model), (context,), out / "lm_context.onnx", input_names=["context"], output_names=["past_k", "past_v", "first"],
                      dynamic_axes={"context": {0: "lc"}, "past_k": {2: "lc"}, "past_v": {2: "lc"}}, opset_version=17, dynamo=False)
    torch.onnx.export(LmKeystrokeGraph(model), (past_k, past_v, first, candidates, lengths), out / "lm_keystroke.onnx",
                      input_names=["past_k", "past_v", "first", "candidates", "lengths"], output_names=["scores"],
                      dynamic_axes={"past_k": {2: "lc"}, "past_v": {2: "lc"}, "candidates": {0: "b", 1: "lb"}, "lengths": {0: "b"}, "scores": {0: "b"}}, opset_version=17, dynamo=False)
    for name in ("lm_context", "lm_keystroke"):
        quantize_dynamic(out / f"{name}.onnx", out / f"{name}.int8.onnx", weight_type=QuantType.QInt8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--strip-context", action="store_true", help="score only the cases that have a context, without it (the eval set's context-stripped subset)")
    parser.add_argument("--lm-only", action="store_true", help="score as a character LM (lm_context / lm_score)")
    args = parser.parse_args()

    run = Path(args.run)
    model, vocab, ckpt = load(run)
    onnx_dir = run / "onnx"
    prefix = "lm_" if args.lm_only else ""
    if not (onnx_dir / f"{prefix}keystroke.int8.onnx").exists():
        onnx_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            (export_lm if args.lm_only else export)(model, onnx_dir)
    ctx, key = session(onnx_dir / f"{prefix}context.int8.onnx", args.threads), session(onnx_dir / f"{prefix}keystroke.int8.onnx", args.threads)
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
            if args.lm_only:
                k, v, first = ctx.run(None, {"context": np.concatenate([[END], context]).astype(np.int64)})
                bodies = [encoder.chars(c) for c in case["candidates"]]
                candidates = np.zeros((len(bodies), max(len(b) for b in bodies)), dtype=np.int64)
                for i, b in enumerate(bodies):
                    candidates[i, : len(b)] = b
                feed = {"past_k": k, "past_v": v, "first": first, "candidates": candidates, "lengths": np.array([len(b) for b in bodies], dtype=np.int64)}
            elif len(context):
                k, v = ctx.run(None, {"context": context})
            else:
                if empty is None:
                    k0, v0 = ctx.run(None, {"context": np.array([END], dtype=np.int64)})
                    empty = (k0[:, :, :0], v0[:, :, :0])
                k, v = empty
            if not args.lm_only:
                letters = np.array(letter_ids(case["input"])[:LETTERS_MAX], dtype=np.int64)
                bodies = [encoder.chars(c) + [END] for c in case["candidates"]]
                width = max(len(b) for b in bodies)
                candidates = np.zeros((len(bodies), width), dtype=np.int64)
                for i, b in enumerate(bodies):
                    candidates[i, : len(b)] = b
                ends = np.array([len(b) - 1 for b in bodies], dtype=np.int64)
                feed = {"past_k": k, "past_v": v, "letters": letters, "candidates": candidates, "ends": ends}
            started = time.perf_counter()
            scores = key.run(None, feed)[0]
            ms = (time.perf_counter() - started) * 1000
            pick = int(np.argmax(scores))
            n += 1
            hits += case["candidates"][pick] == case["gold"]
            out.write(json.dumps({"id": case["id"], "pick": pick, "scores": [float(s) for s in scores], "ms": round(ms, 3)}) + "\n")
    print(f"{hits}/{n} hits ({100 * hits / max(1, n):.1f}%), checkpoint step {ckpt.get('step')}, dev hits {ckpt.get('dev_hits')}", file=sys.stderr)


if __name__ == "__main__":
    main()
