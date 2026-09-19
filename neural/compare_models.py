"""Compares models case by case, which is the only way a small evaluation set decides anything.

`rerank_eval.py` reports totals. Across the 56 comparable cases in `eval/sentences-v1.tsv` a spread
of three is well inside the noise for a single arm, so totals cannot separate "this shape is
genuinely better" from "this shape won three coin flips". What separates them is whether the models
agree on *which* cases they get right: when one model's correct set is a strict superset of
another's, every case the second solves the first solves too, and nothing argues for the second.

That is what settled the shape sweep. Five shapes sat within one case of each other on totals; case
by case, none ever solved a case the cheapest one missed, so nothing argued for paying more. A
larger model then turned out to be a strict superset of all of them, solving the two cases they all
missed — a different kind of result from "scored two higher", and only this view shows it.

Recovered and broken are reported separately for the same reason. Two models can recover the same
number of cases from the engine and still differ in total because one of them breaks cases the
engine already had right, and breaking a case the engine got right is not what a user experiences
as "one fewer improvement".

Reads the files `rerank_eval.py --per-case` writes, so the scoring and gating rules live in exactly
one place. An earlier version of this tool copied them, which would have drifted the first time
either was touched.

usage:
  python neural/rerank_eval.py --model a.safetensors --cases dumps/sentences.jsonl --per-case a.jsonl
  python neural/rerank_eval.py --model b.safetensors --cases dumps/sentences.jsonl --per-case b.jsonl
  python neural/compare_models.py a.jsonl b.jsonl
"""

import argparse
import json
import os


def load(path, bucket):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if bucket and row["bucket"] != bucket:
                continue
            rows[row["id"]] = row
    if not rows:
        raise SystemExit(f"{path}: no cases in bucket {bucket!r}")
    return rows


def by_size(right):
    return sorted(right.items(), key=lambda item: -len(item[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="per-case files, one per model")
    parser.add_argument(
        "--bucket",
        default="decoded",
        choices=["decoded", "dictionary", "all"],
        help="dictionary hits are gated away from the model and cannot separate two models, "
        "so they are excluded by default",
    )
    args = parser.parse_args()
    bucket = None if args.bucket == "all" else args.bucket

    models = {}
    for path in args.files:
        models[os.path.splitext(os.path.basename(path))[0]] = load(path, bucket)

    ids = set.intersection(*(set(rows) for rows in models.values()))
    if not ids:
        raise SystemExit("these files share no case ids; were they run against the same dump?")
    for name, rows in models.items():
        if set(rows) != ids:
            print(f"note: {name} covers a different set of cases; comparing the {len(ids)} in common")

    sample = next(iter(models.values()))
    engine = {case for case in ids if sample[case]["engine"]}
    reachable = {case for case in ids if sample[case]["reachable"]}
    right = {name: {case for case in ids if rows[case]["reranked"]} for name, rows in models.items()}

    print(f"{len(ids)} comparable cases")
    print(f"  {'engine already correct':<34}{len(engine)}")
    print(f"  {'gold present among candidates':<34}{len(reachable)}   <- ceiling for any reranker")
    for name, correct in by_size(right):
        broke = len(engine - correct)
        note = "" if not broke else f", broke {broke}"
        print(f"  {name:<34}{len(correct)}   recovered {len(correct - engine)}{note}")

    unreachable = ids - reachable
    if unreachable:
        print(f"\ngold absent from the candidate list in {len(unreachable)} cases — candidate")
        print("generation, not the model:")
        print("  " + ", ".join(sorted(unreachable)))

    everyone = set().union(*right.values())
    missed = reachable - everyone
    if missed:
        print(f"\nreachable but missed by every model ({len(missed)}):")
        print("  " + ", ".join(sorted(missed)))

    print("\nsolved by one model alone:")
    for name, correct in by_size(right):
        others = set().union(*(v for k, v in right.items() if k != name)) if len(right) > 1 else set()
        unique = sorted(correct - others)
        print(f"  {name:<34}{unique if unique else 'none'}")

    if len(right) > 1:
        print("\npairwise (row wins / column wins / both / neither):")
        names = [name for name, _ in by_size(right)]
        for index, a in enumerate(names):
            for b in names[index + 1 :]:
                ra, rb = right[a], right[b]
                verdict = ""
                if ra > rb:
                    verdict = f"   <- {a} is a strict superset"
                elif rb > ra:
                    verdict = f"   <- {b} is a strict superset"
                print(
                    f"  {a:<26} vs {b:<26} "
                    f"{len(ra - rb)} / {len(rb - ra)} / {len(ra & rb)} / {len(ids - ra - rb)}{verdict}"
                )


if __name__ == "__main__":
    main()
