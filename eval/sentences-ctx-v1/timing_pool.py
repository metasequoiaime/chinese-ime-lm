"""Draw the fixed subset of `pool.jsonl` that every scorer is timed on (`pool-timing.jsonl`).

Timing every scorer on all 6317 cases, several passes each, would take hours and tell little more than a few hundred cases do. The subset is stratified by what changes the cost of a decision: level (clause or word), typing style, and whether the case has a context. Each of the eight strata gives the same number of cases, drawn with a seeded generator, and the cases keep `pool.jsonl`'s order, so the file is the same on every run and for every scorer. Because the strata are equal in size and not in proportion to the pool, a time over the whole subset is not a time over the pool: `summarize.py` reports times per subset.

usage:
  python eval/sentences-ctx-v1/timing_pool.py [--pool eval/sentences-ctx-v1/pool.jsonl] [--out eval/sentences-ctx-v1/pool-timing.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = "sentences-ctx-v1-timing"
PER_STRATUM = 60


def stratum(case):
    return case["level"], case["style"], bool(case["context"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default=str(HERE / "pool.jsonl"))
    parser.add_argument("--out", default=str(HERE / "pool-timing.jsonl"))
    parser.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    args = parser.parse_args()

    with open(args.pool, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    cases = [json.loads(line) for line in lines]
    strata = {}
    for index, case in enumerate(cases):
        strata.setdefault(stratum(case), []).append(index)
    chosen = set()
    for key in sorted(strata):
        members = strata[key]
        rng = random.Random(f"{SEED}-{key[0]}-{key[1]}-{'context' if key[2] else 'none'}")
        if len(members) < args.per_stratum:
            raise SystemExit(f"{key}: {len(members)} cases, fewer than {args.per_stratum}")
        chosen.update(rng.sample(members, args.per_stratum))
    with open(args.out, "w", encoding="utf-8") as out:
        for index in sorted(chosen):
            out.write(lines[index] if lines[index].endswith("\n") else lines[index] + "\n")
    print(f"{len(chosen)} cases, {args.per_stratum} from each of {len(strata)} strata -> {args.out}")


if __name__ == "__main__":
    main()
