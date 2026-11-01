"""Export the LM-only keystroke path as the single graph an input method runs (`ChoiceModel.lm_step`), in ONNX Runtime's ORT format for a minimal runtime build.

Writes into `--out`:
- `lm_step.onnx`: the graph in fp32;
- `lm_step.ort`: weights quantized dynamically to uint8 and the graph optimized ahead of time ("Fixed" style), which is what a minimal ONNX Runtime build loads. uint8 rather than int8 weights because ONNX Runtime picks its int8 kernels by CPU feature: with int8 weights an Apple M4 (I8MM) and an M1 (no I8MM) disagreed on 42 of 6317 ctx-v1 pages, with uint8 weights they give bit-identical scores at the same accuracy;
- `required_operators_and_types.config`: the operators and types the model uses, the input to ONNX Runtime's `--include_ops_by_config` and `--enable_reduced_operator_type_support`;
- `vocab.json`: the run's token list (single characters are the model's characters; ids are list positions);
- `model.json`: what a runtime needs besides the graph: the cache shape, the context it was trained on, and whether candidate characters may be read without the softmax.

With `--golden POOL`, also types the first `--golden-cases` cases of a pool file with simulated pages (each case's candidates cut to 1, 2, ... characters, as `step_inputs` documents) through the ORT file with ONNX Runtime's Python package and writes every keystroke's candidates and scores to `golden.jsonl`, for a runtime's own tests to replay.

With `--random-shape LAYERSxWIDTH --random-chars TEXT` it exports a randomly initialised model over just those characters instead of `--run`, small enough to commit as a test fixture.

A trained model needs `--corpus-dir`, the directories whose `*.source.json` records describe every corpus the model learned from (directly, through its initialisation, or through a teacher it distilled). The attribution is built from those records as `neural/export.py` builds it, refused if any source is share-alike, and written both into the graph's metadata (`attribution`, `license`), which the ORT file keeps, and into `model.json`: the corpora's licences require the attribution to travel with the weights.

usage:
  python neural/choice/export_step.py --run <lm run dir> --corpus-dir <dir with *.source.json> --out <dir>
  python neural/choice/export_step.py --random-shape 2x32 --random-chars 今天我们去北京背景玩 --out <dir> --golden <pool.jsonl> --golden-cases 40
"""

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import END, RESERVED, ChoiceConfig, ChoiceModel, LmStepGraph, context_inputs, step_inputs

CONTEXT_CHARS = 64
NAMES = ["cache_k", "cache_v", "cache_h", "cache_z", "nodes", "positions", "visible", "candidates", "steps", "lengths"]
OUTPUTS = ["scores", "next_k", "next_v", "next_h", "next_z"]


def empty_cache(cfg):
    d = cfg.n_embd // cfg.n_head
    return {
        "cache_k": np.zeros((cfg.n_layer, cfg.n_head, 0, d), np.float32),
        "cache_v": np.zeros((cfg.n_layer, cfg.n_head, 0, d), np.float32),
        "cache_h": np.zeros((0, cfg.n_embd), np.float32),
        "cache_z": np.zeros((0,), np.float32),
    }


def padded(bodies):
    width = max(len(b) for b in bodies)
    out = np.zeros((len(bodies), width), np.int64)
    for i, b in enumerate(bodies):
        out[i, : len(b)] = b
    return out


def context_feed(context):
    nodes, positions, visible, candidates, steps, lengths = context_inputs(context)
    return {
        "nodes": np.array(nodes, np.int64),
        "positions": np.array(positions, np.int64),
        "visible": np.array(visible, bool),
        "candidates": np.array(candidates, np.int64),
        "steps": np.array(steps, np.int64),
        "lengths": np.array(lengths, np.int64),
    }


def keystroke_feed(n_context, bodies, cached):
    nodes, positions, visible, steps, every = step_inputs(n_context, bodies, cached)
    feed = {
        "nodes": np.array(nodes, np.int64),
        "positions": np.array(positions, np.int64),
        "visible": np.array(visible, bool).reshape(len(nodes), n_context + len(every)),
        "candidates": padded(bodies),
        "steps": np.array(steps, np.int64),
        "lengths": np.array([len(b) for b in bodies], np.int64),
    }
    return feed, every


def pages(candidates):
    """The simulated keystroke pages of one case: its candidates cut to 1, 2, ... characters, first occurrence kept."""
    out = []
    for k in range(1, max(len(c) for c in candidates) + 1):
        page = []
        for c in candidates:
            if c[:k] not in page:
                page.append(c[:k])
        out.append(page)
    return out


LICENSE = "Apache-2.0"


def attribution(directories):
    """The attribution for the corpora recorded in `directories`, in `neural/export.py`'s wording; stops on a share-alike source."""
    records = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.source.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit(
            f"no *.source.json in {', '.join(directories)}: the corpus is unrecorded, so the attribution cannot be derived"
        )
    if any(r["share_alike"] for r in records):
        raise SystemExit(
            "a corpus imposes a share-alike obligation; these weights are not eligible for release under this project's policy"
        )
    named = list(dict.fromkeys(f"{r['name']} ({r['license']}, {r['url']})" for r in records))
    listed = ", ".join(named[:-1]) + " and " + named[-1] if len(named) > 1 else named[0]
    return f"Trained on {listed}."


def stamp(path, text):
    """Write the attribution and licence into an ONNX file's metadata, which quantization and the ORT conversion keep."""
    import onnx

    graph = onnx.load(str(path))
    del graph.metadata_props[:]
    for key, value in (("attribution", text), ("license", LICENSE)):
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(graph, str(path))


def export(model, out, credit=None):
    cfg = model.cfg
    torch.manual_seed(0)
    context = [END, 3, 4, 5]
    cache = {n: torch.from_numpy(a) for n, a in empty_cache(cfg).items()}
    feed = {n: torch.from_numpy(a) for n, a in context_feed(context).items()}
    with torch.no_grad():
        _, k, v, h, z = model.lm_step(*[(cache | feed)[n] for n in NAMES])
        bodies = [[6, 7, 8], [6, 9]]
        keys, _ = keystroke_feed(len(context), bodies, [])
        example = {"cache_k": k, "cache_v": v, "cache_h": h, "cache_z": z} | {
            n: torch.from_numpy(a) for n, a in keys.items()
        }
        dynamic = {
            "cache_k": {2: "m"},
            "cache_v": {2: "m"},
            "cache_h": {0: "m"},
            "cache_z": {0: "m"},
            "nodes": {0: "n"},
            "positions": {0: "n"},
            "visible": {0: "n", 1: "mn"},
            "candidates": {0: "b", 1: "lb"},
            "steps": {0: "b", 1: "lb"},
            "lengths": {0: "b"},
            "scores": {0: "b"},
            "next_k": {2: "mn"},
            "next_v": {2: "mn"},
            "next_h": {0: "mn"},
            "next_z": {0: "mn"},
        }
        torch.onnx.export(
            LmStepGraph(model),
            tuple(example[n] for n in NAMES),
            out / "lm_step.onnx",
            input_names=NAMES,
            output_names=OUTPUTS,
            dynamic_axes=dynamic,
            opset_version=17,
            dynamo=False,
        )
    if credit:
        stamp(out / "lm_step.onnx", credit)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantized = out / "ort-work"
    quantized.mkdir(exist_ok=True)
    quantize_dynamic(out / "lm_step.onnx", quantized / "lm_step.onnx", weight_type=QuantType.QUInt8)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(quantized),
            "--optimization_style",
            "Fixed",
            "--enable_type_reduction",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    (quantized / "lm_step.ort").replace(out / "lm_step.ort")
    (quantized / "required_operators_and_types.config").replace(out / "required_operators_and_types.config")
    for leftover in quantized.iterdir():
        leftover.unlink()
    quantized.rmdir()


def write_golden(cfg, tokens, out, pool, count):
    import onnxruntime as ort

    ids = {t: i for i, t in enumerate(tokens) if len(t) == 1}

    def encode(text):
        return [ids.get(c, 1) for c in text]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(out / "lm_step.ort"), options)
    with open(pool, encoding="utf-8") as handle, open(out / "golden.jsonl", "w", encoding="utf-8") as golden:
        for line in list(handle)[:count]:
            case = json.loads(line)
            text = case.get("context", "")[-CONTEXT_CHARS:]
            context = [END, *encode(text)]
            result = session.run(None, empty_cache(cfg) | context_feed(context))
            cache = dict(zip(NAMES[:4], result[1:], strict=True))
            cached = []
            keystrokes = []
            for page in pages(case["candidates"]):
                feed, cached = keystroke_feed(len(context), [encode(c) for c in page], cached)
                result = session.run(None, cache | feed)
                cache = dict(zip(NAMES[:4], result[1:], strict=True))
                keystrokes.append({"candidates": page, "scores": [float(s) for s in result[0]]})
            golden.write(
                json.dumps({"id": case["id"], "context": text, "keystrokes": keystrokes}, ensure_ascii=False) + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="an LM run directory: checkpoint.pt and vocab.json")
    parser.add_argument("--random-shape", help="LAYERSxWIDTH of a randomly initialised model, instead of --run")
    parser.add_argument("--random-chars", default="", help="with --random-shape: the model's characters")
    parser.add_argument(
        "--self-normalized", action="store_true", help="with --random-shape: mark the model self_normalized"
    )
    parser.add_argument(
        "--corpus-dir",
        action="append",
        default=[],
        help="with --run: a directory of *.source.json records of the corpora the model learned from; repeat for several",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--golden", help="a pool file whose first cases to replay through the exported model")
    parser.add_argument("--golden-cases", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.random_shape:
        layers, width = (int(x) for x in args.random_shape.split("x"))
        tokens = RESERVED + list(dict.fromkeys(args.random_chars))
        cfg = ChoiceConfig(
            vocab=len(tokens),
            n_layer=layers,
            n_head=max(1, width // 16),
            n_embd=width,
            max_positions=128,
            self_normalized=args.self_normalized,
            gelu_tanh=False,
        )
        torch.manual_seed(20261101)
        model = ChoiceModel(cfg).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0.0, 0.3)
    else:
        run = Path(args.run)
        state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        cfg = ChoiceConfig(**state["config"])
        model = ChoiceModel(cfg).eval()
        model.load_state_dict(state["model"])
        tokens = json.loads((run / "vocab.json").read_text(encoding="utf-8"))["tokens"]
    credit = None if args.random_shape else attribution(args.corpus_dir)
    export(model, out, credit)
    (out / "vocab.json").write_text(json.dumps({"tokens": tokens}, ensure_ascii=False), encoding="utf-8")
    meta = {
        "layers": cfg.n_layer,
        "heads": cfg.n_head,
        "head_dim": cfg.n_embd // cfg.n_head,
        "width": cfg.n_embd,
        "max_positions": cfg.max_positions,
        "context_chars": CONTEXT_CHARS,
        "bos": END,
        "unk": 1,
        "self_normalized": cfg.self_normalized,
        "attribution": credit,
        "license": None if credit is None else LICENSE,
        "config": dataclasses.asdict(cfg),
    }
    (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.golden:
        write_golden(cfg, tokens, out, args.golden, args.golden_cases)
    print({p.name: p.stat().st_size for p in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
