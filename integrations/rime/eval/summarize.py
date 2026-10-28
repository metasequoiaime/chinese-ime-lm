"""Compares rime_eval outputs case by case.

usage: summarize.py <baseline.tsv> <candidate.tsv> [<more.tsv> ...]

Every file after the first is compared with the first. Rescues and breakages are reported separately, as docs/measurements.md does, because a single total hides a model that fixes some cases by breaking others. The McNemar exact test is over the cases where the two runs disagree; at n=60 it is the honest way to say whether a difference is distinguishable from noise.
"""

import csv
import math
import sys
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def mcnemar_exact(wins, losses):
    """Two-sided exact binomial p-value for `wins` against `losses` among discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def correct(row):
    return row["top"] == row["gold"]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = [(Path(path).stem, load(path)) for path in sys.argv[1:]]
    base_name, base = runs[0]
    ids = list(base)
    print(f"cases: {len(ids)}")
    reachable = sum(1 for case in ids if int(base[case]["gold_rank"]) >= 0)
    for name, run in runs:
        top1 = sum(1 for case in ids if correct(run[case]))
        present = sum(1 for case in ids if int(run[case]["gold_rank"]) >= 0)
        print(f"{name}: top-1 {top1}/{len(ids)} ({top1 / len(ids):.3f}), gold among first 32: {present}")
    print(f"(gold reachable in {base_name}: {reachable})")

    for name, run in runs[1:]:
        rescued = [case for case in ids if correct(run[case]) and not correct(base[case])]
        broken = [case for case in ids if correct(base[case]) and not correct(run[case])]
        changed = [case for case in ids if run[case]["top"] != base[case]["top"]]
        print()
        print(f"{name} against {base_name}: {len(rescued)} rescued, {len(broken)} broken, {len(changed)} first candidates changed, McNemar p = {mcnemar_exact(len(rescued), len(broken)):.3f}")
        for case in changed:
            mark = "+" if case in rescued else "-" if case in broken else " "
            print(f"  {mark} {case} gold {base[case]['gold']}: {base[case]['top']} -> {run[case]['top']}")


if __name__ == "__main__":
    main()
