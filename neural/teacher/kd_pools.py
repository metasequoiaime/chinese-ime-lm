"""Write the training pools the teacher scores for distillation: the non-spam pools that come after the ones its LoRA was trained on.

`lora.py --examples N --drop-spam` trains on the first N training pools that pass the spam filter, so this skips that many passing pools and writes the next `--count` passing ones, in the file's (shuffled) order. Pools the teacher was fitted on would give the student the teacher's training-set confidence, not the confidence it has on text it has not seen.

usage:
  python neural/teacher/kd_pools.py --train <data>/pools/train.jsonl --skip 40000 --count 100000 --out <data>/pools/kd.jsonl
"""

import argparse
import json

from lora import SPAM


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--skip", type=int, required=True, help="the --examples the teacher's LoRA was trained with")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    passed = written = spam = 0
    with open(args.train, encoding="utf-8") as source, open(args.out, "w", encoding="utf-8") as out:
        for line in source:
            case = json.loads(line)
            if any(word in case["context"] or word in case["gold"] for word in SPAM):
                spam += 1
                continue
            passed += 1
            if passed <= args.skip:
                continue
            out.write(line)
            written += 1
            if written == args.count:
                break
    print(f"skipped {args.skip} passing pools, wrote {written}, {spam} spam pools left out on the way")
    if written < args.count:
        raise SystemExit(f"only {written} pools after the first {args.skip}")


if __name__ == "__main__":
    main()
