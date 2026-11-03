"""Average a fine-tuned choice model with the checkpoint it started from: `base + alpha * (tuned - base)`, tensor by tensor.

Training the LM-only keystroke models further on the chat pools (`train.py --phase pools --init <released run>`) teaches them the chat but costs a few tenths of a point on ctx-v1, and that cost did not go away by mixing general pools into the training or by anchoring the corpus windows to the starting model. Moving only part of the way from the released weights towards the tuned ones kept most of the chat gain and gave the ctx-v1 points back: on 2026-11-04 the released 8x448 took `alpha` 0.7 and the 6x256 0.5, which kept ctx-v1 with a context at 91.4% and 90.1% on the exported `lm_step.ort` (the 6x256 at 0.7 had 89.96%, three cases short of the 90% bar).

Both checkpoints must have the same configuration (a missing `self_normalized` counts as false, as `ChoiceConfig` reads it). The output keeps the tuned checkpoint's other fields and records where it came from under `interpolated`, so `evaluate.py` and `export_step.py` take it like any run: put it in a directory with the run's `vocab.json` and `config.json`.

usage:
  python neural/choice/interpolate.py --base <released run>/checkpoint.pt --tuned <tuned run>/checkpoint.pt --alpha 0.7 --out <dir>/checkpoint.pt
"""

import argparse

import torch


def same_config(a, b):
    def norm(config):
        return {**config, "self_normalized": bool(config.get("self_normalized", False))}

    return norm(a) == norm(b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="0 gives the base, 1 the tuned model")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=True)
    if not same_config(base["config"], tuned["config"]):
        raise SystemExit(f"the two checkpoints differ in configuration: {base['config']} vs {tuned['config']}")
    if base["model"].keys() != tuned["model"].keys():
        raise SystemExit("the two checkpoints hold different tensors")
    mixed = {}
    for name, value in tuned["model"].items():
        if value.is_floating_point():
            start = base["model"][name].float()
            mixed[name] = (start + args.alpha * (value.float() - start)).to(value.dtype)
        else:
            mixed[name] = value
    torch.save(
        {**tuned, "model": mixed, "interpolated": {"base": args.base, "tuned": args.tuned, "alpha": args.alpha}},
        args.out,
    )


if __name__ == "__main__":
    main()
