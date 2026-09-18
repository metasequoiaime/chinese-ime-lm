"""Export a trained checkpoint as a single file the client can ship and load.

The output is plain safetensors. The configuration, the vocabulary and the corpus attribution travel in the safetensors `__metadata__` header, so the model is one self-describing file with no sidecars to keep in sync and no bespoke container format to parse on the other side.

Quantization is per-output-row and symmetric. Norm parameters and biases stay in float32: they are a negligible share of the file and the most sensitive to rounding.

usage:
  python export.py --run runs/keyboard --out dist/sentence-v1.safetensors [--precision f16|int8]
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

from model import CharLM, Config


def attribution(run):
    """The corpora this run was actually trained on, named from their own records.

    This used to be a constant listing every source the pipeline can fetch, which meant every
    exported model claimed to contain Chinese Wikipedia and MDN whether or not it had ever seen
    them. The field exists so that a redistributor can rely on it — `SECURITY.md` and `NOTICE` both
    require it to travel with the weights — so a value that is right only by coincidence is worse
    than no value at all: a model trained purely on permissive text was being stamped share-alike,
    and a genuinely share-alike model would have been stamped identically.

    Derived rather than declared, and it refuses to guess: a corpus file with no record beside it
    stops the export. Guessing is what produced the bug.
    """
    config = os.path.join(run, "config.json")
    if not os.path.exists(config):
        raise SystemExit(
            f"{config} is missing, so the corpus this run used is unrecorded and the attribution "
            f"cannot be derived. Retrain with the current train.py."
        )
    with open(config, encoding="utf-8") as handle:
        corpora = json.load(handle)["corpus"]

    records = []
    for path in corpora:
        beside = f"{path}.source.json"
        if not os.path.exists(beside):
            raise SystemExit(
                f"{beside} is missing, so what {path} contains is unknown. Re-fetch it with "
                f"corpus/fetch.py, which writes that record."
            )
        with open(beside, encoding="utf-8") as handle:
            records.append(json.load(handle))

    named = [f"{r['name']} ({r['license']}, {r['url']})" for r in records]
    listed = ", ".join(named[:-1]) + " and " + named[-1] if len(named) > 1 else named[0]
    text = f"Trained on {listed}."

    # Stated in the file rather than left for a reader to work out from the licence names, because
    # this is the one bit an adopter has to check before shipping.
    if any(r["share_alike"] for r in records):
        text += (
            " At least one of these imposes a share-alike obligation, so these weights are not "
            "eligible for release under this project's policy."
        )
    return text, records


# The weights, not the scripts that made them. `reference/` is Apache-2.0 for the same reason this
# is: a great many open-source input methods are MIT, Apache or BSD and cannot link GPL code, and a
# public resource nobody can adopt is not one. Override with --license if you trained your own on
# something that obliges otherwise.
DEFAULT_LICENSE = "Apache-2.0"

# Rounding these to int8 costs more accuracy than it saves bytes.
KEEP_FLOAT = ("ln1.", "ln2.", "ln_f.", ".bias", "pos.weight")

# The name the client installs the model under, and the name the host looks for.
ARTIFACT_NAME = "sentence-model.safetensors"


def quantize_int8(tensor):
    """Symmetric per-row quantization. Returns the int8 values and the float32 scale per row."""
    flat = tensor.reshape(tensor.shape[0], -1).float()
    scale = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    values = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return values.reshape(tensor.shape), scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="training output directory containing checkpoint.pt and vocab.json"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", choices=["f16", "int8"], default="f16")
    parser.add_argument("--url", default="", help="published location, for the resource lock entry")
    # Was hardcoded to the repository's own GPL-3.0, which is wrong twice. README.md says outright
    # that released weights are not governed by the code licence, and GPL-3.0 is share-alike — so
    # every exported model carried an obligation that this project spends its corpus policy avoiding,
    # for the sake of adopters who should not need a legal review to use it. What the weights are
    # licensed under is the publisher's decision, so it is asked for rather than assumed.
    parser.add_argument(
        "--license",
        default=DEFAULT_LICENSE,
        help=f"SPDX identifier written into the model metadata (default {DEFAULT_LICENSE})",
    )
    args = parser.parse_args()

    # Before the weights are read: an export that cannot say what it was trained on should fail
    # while it has produced nothing, not after it has written a file someone might ship.
    attribution_text, corpus_records = attribution(args.run)

    checkpoint = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location="cpu", weights_only=True)
    with open(os.path.join(args.run, "vocab.json"), encoding="utf-8") as handle:
        vocab = json.load(handle)["tokens"]
    cfg = Config(**checkpoint["config"])

    model = CharLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tensors = {}
    for name, tensor in model.state_dict().items():
        # The head is tied to the token table; writing it again would double the largest tensor.
        if name == "head.weight":
            continue
        if args.precision == "int8" and tensor.dim() == 2 and not any(k in name for k in KEEP_FLOAT):
            values, scale = quantize_int8(tensor)
            tensors[name] = values.contiguous()
            tensors[name + ".scale"] = scale.contiguous()
        elif args.precision == "f16" and tensor.is_floating_point() and not any(k in name for k in KEEP_FLOAT):
            tensors[name] = tensor.to(torch.float16).contiguous()
        else:
            tensors[name] = tensor.float().contiguous()

    metadata = {
        "format": "chinese-ime-lm",
        "version": "1",
        "precision": args.precision,
        "config": json.dumps(checkpoint["config"], separators=(",", ":")),
        "vocab": json.dumps(vocab, ensure_ascii=False, separators=(",", ":")),
        "tied_embeddings": "true",
        "validation_loss": f"{checkpoint['val']:.6f}",
        "training_steps": str(checkpoint["step"]),
        "license": args.license,
        "attribution": attribution_text,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)
    size = os.path.getsize(args.out)
    print(
        f"{args.out}: {size / 1e6:.1f} MB, {cfg.parameters():,} parameters, {args.precision}, validation loss {checkpoint['val']:.4f}"
    )
    sources = ", ".join(f"{r['source']} ({r['license']})" for r in corpus_records)
    blocked = [r["source"] for r in corpus_records if r["share_alike"]]
    if blocked:
        print(f"corpus: {sources}", file=sys.stderr)
        print(
            f"NOT RELEASABLE: {', '.join(blocked)} imposes share-alike. These weights are usable "
            f"locally but may not be published under this project's policy.",
            file=sys.stderr,
        )
    else:
        print(f"corpus: {sources} — no share-alike obligation, releasable", file=sys.stderr)

    # The entry a resource lock needs. A model is installed and verified exactly like a dictionary,
    # by name, length and digest, so publishing one means adding this to a lock rather than copying
    # a file into place by hand.
    digest = hashlib.sha256()
    with open(args.out, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    entry = {
        "name": ARTIFACT_NAME,
        "url": args.url or f"https://example.invalid/{ARTIFACT_NAME}",
        "sha256": digest.hexdigest(),
        "size": size,
    }
    print("\nresource lock entry:")
    print(json.dumps(entry, indent=2))
    if not args.url:
        print("(pass --url once the file is published; the placeholder is not a valid source)")


if __name__ == "__main__":
    main()
