"""P0 of `docs/choice-model.md`: how large a choice model fits the keystroke budget (p95 <= 10 ms) on ONNX Runtime.

For each shape, a randomly initialised `ChoiceModel` is exported as its two graphs, quantized to int8 the way ONNX Runtime's dynamic quantization does it, and timed on real pages: the cases of a pool file give the context (its last 64 characters), the letters typed and the candidates, so sequence lengths and page sizes are the real ones. The keystroke graph is timed with the context already encoded (it is encoded when text is committed); the context graph is timed on its own. Every timed run is kept as its own sample, not a median of repeats, because the budget is on the tail (p95, p99 and the slowest run): an input method that is fast on average and stalls now and then is a slow input method. Thread counts are given with `--threads` (intra-op threads, ONNX Runtime's default spin-waiting).

Weights are random, so this measures time and size only; accuracy comes from P2.

usage:
  python neural/choice/latency.py --pool <pools/ctx-v1.jsonl> --out <dir> [--shape 6x256 --shape 8x384 ...] [--threads 1 2 4] [--cases 600]
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import END, ChoiceConfig, ChoiceModel, ContextGraph, KeystrokeGraph, letter_ids  # noqa: E402

CONTEXT_CHARS = 64


def page(case, rng, vocab):
    """Character ids are random (the weights are too); only the lengths come from the case."""
    ids = lambda text: [rng.randrange(3, vocab) for _ in text]
    context = ids(case.get("context", "")[-CONTEXT_CHARS:])
    letters = letter_ids(case["input"])
    rows = [ids(c) + [END] for c in case["candidates"]]
    width = max(len(r) for r in rows)
    candidates = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        candidates[i, : len(r)] = r
    ends = np.array([len(r) - 1 for r in rows], dtype=np.int64)
    return np.array(context, dtype=np.int64), np.array(letters, dtype=np.int64), candidates, ends


def export(model, out):
    cfg = model.cfg
    context = torch.randint(3, cfg.vocab, (12,))
    past_k, past_v = model.encode_context(context)
    letters = torch.tensor(letter_ids("nihao"))
    candidates = torch.randint(3, cfg.vocab, (3, 5))
    ends = torch.tensor([4, 3, 2])
    torch.onnx.export(ContextGraph(model), (context,), out / "context.onnx", input_names=["context"], output_names=["past_k", "past_v"],
                      dynamic_axes={"context": {0: "lc"}, "past_k": {2: "lc"}, "past_v": {2: "lc"}}, opset_version=17, dynamo=False)
    torch.onnx.export(KeystrokeGraph(model), (past_k, past_v, letters, candidates, ends), out / "keystroke.onnx",
                      input_names=["past_k", "past_v", "letters", "candidates", "ends"], output_names=["scores"],
                      dynamic_axes={"past_k": {2: "lc"}, "past_v": {2: "lc"}, "letters": {0: "la"}, "candidates": {0: "b", 1: "lb"}, "ends": {0: "b"}, "scores": {0: "b"}},
                      opset_version=17, dynamo=False)
    for name in ("context", "keystroke"):
        quantize_dynamic(out / f"{name}.onnx", out / f"{name}.int8.onnx", weight_type=QuantType.QInt8)


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def check(model, out, pages):
    """The int8 graphs against PyTorch on a few pages: the picks should agree and the scores be close."""
    ctx, key = session(out / "context.int8.onnx", 1), session(out / "keystroke.int8.onnx", 1)
    worst, same = 0.0, 0
    for context, letters, candidates, ends in pages[:20]:
        if len(context) == 0:
            continue
        with torch.no_grad():
            pk, pv = model.encode_context(torch.from_numpy(context))
            want = model.score(pk, pv, torch.from_numpy(letters), torch.from_numpy(candidates), torch.from_numpy(ends)).numpy()
        k, v = ctx.run(None, {"context": context})
        got = key.run(None, {"past_k": k, "past_v": v, "letters": letters, "candidates": candidates, "ends": ends})[0]
        worst = max(worst, float(np.abs(got - want).max()))
        same += int(got.argmax() == want.argmax())
    return worst, same


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def bench(out, pages, threads, repeats):
    ctx, key = session(out / "context.int8.onnx", threads), session(out / "keystroke.int8.onnx", threads)
    lc_empty = None
    keystroke, context_ms = [], []
    for context, letters, candidates, ends in pages:
        if len(context):
            t = time.perf_counter()
            k, v = ctx.run(None, {"context": context})
            context_ms.append((time.perf_counter() - t) * 1000)
        else:
            if lc_empty is None:
                k0, v0 = ctx.run(None, {"context": np.array([3], dtype=np.int64)})
                lc_empty = (k0[:, :, :0], v0[:, :, :0])
            k, v = lc_empty
        feed = {"past_k": k, "past_v": v, "letters": letters, "candidates": candidates, "ends": ends}
        key.run(None, feed)  # warm
        for _ in range(repeats):
            t = time.perf_counter()
            key.run(None, feed)
            keystroke.append((time.perf_counter() - t) * 1000)
    tail = lambda xs: {"p50": percentile(xs, 0.5), "p95": percentile(xs, 0.95), "p99": percentile(xs, 0.99), "max": max(xs)}
    return {"threads": threads, "samples": len(keystroke), "keystroke_ms": tail(keystroke), "context_ms": tail(context_ms) if context_ms else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shape", action="append", help="LAYERSxWIDTH, e.g. 6x256")
    parser.add_argument("--heads", type=int, default=0, help="0: width / 64")
    parser.add_argument("--vocab", type=int, default=12288)
    parser.add_argument("--cases", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    shapes = args.shape or ["4x256", "6x256", "6x384", "8x384", "12x384", "12x512"]

    rng = random.Random(args.seed)
    with open(args.pool, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    rng.shuffle(cases)
    cases = cases[: args.cases]
    pages = [page(c, rng, args.vocab) for c in cases]
    sizes = {
        "cases": len(pages),
        "candidates_mean": statistics.mean(len(p[3]) for p in pages),
        "candidate_tokens_mean": statistics.mean(int(p[3].sum() + len(p[3])) for p in pages),
        "letters_mean": statistics.mean(len(p[1]) for p in pages),
        "with_context": sum(1 for p in pages if len(p[0])),
    }
    print(json.dumps(sizes), flush=True)

    root = Path(args.out)
    results = []
    for shape in shapes:
        layers, width = (int(x) for x in shape.split("x"))
        cfg = ChoiceConfig(vocab=args.vocab, n_layer=layers, n_embd=width, n_head=args.heads or width // 64)
        torch.manual_seed(args.seed)
        model = ChoiceModel(cfg).eval()
        out = root / shape
        out.mkdir(parents=True, exist_ok=True)
        export(model, out)
        worst, same = check(model, out, pages)
        size_mb = sum((out / f"{n}.int8.onnx").stat().st_size for n in ("context", "keystroke")) / 1e6
        row = {"shape": shape, "params": sum(p.numel() for p in model.parameters()), "non_embedding": model.non_embedding_parameters(), "int8_mb": round(size_mb, 1), "check": {"max_abs_diff": worst, "same_pick_of_20": same}}
        for threads in args.threads:
            row[f"t{threads}"] = bench(out, pages, threads, args.repeats)
        print(json.dumps(row), flush=True)
        results.append(row)
    (root / "latency.json").write_text(json.dumps({"sizes": sizes, "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
