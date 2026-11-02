"""Accuracy baselines on the dev pools: the engine's own order and this project's two released models.

Each model picks one candidate per pool with `reference/examples/pool.rs`, the scorer of the eval sets: the highest mean log-probability per character, the first on a tie, with the context cut the way MSIME hands it to its reranker (`rerank_context`). The engine's order picks the first candidate. A case counts when the pick is the gold. Hits are reported per subset with a Wilson 95% interval, the subsets the context eval set reports (kind of case, with or without context) plus the clause subsets per source.

Only accuracy: `pool` is run with one pass and its timings are discarded.

usage:
  python neural/teacher/score_dev.py --pool <data>/pools/dev.jsonl --pool-bin reference/target/release/examples/pool \\
    --model ours-4m=<dist>/sentence-model.safetensors --model ours-25m=<dist>/sentence-model-desktop.safetensors \\
    --picks-dir <data>/baselines --out <data>/baselines/dev-results.json
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

Z = 1.959963984540054


def wilson(hits, n):
    """The eval set's `summarize.py` interval."""
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    denominator = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denominator
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def subsets(pool):
    """`key -> [case ids]`, in report order."""
    out = {}

    def add(key, test):
        ids = [case["id"] for case in pool if test(case)]
        if ids:
            out[key] = ids

    for level, style in [("clause", "full"), ("word", "full"), ("word", "initials"), ("word", "mixed")]:
        base = f"{level}/{style}"
        add(base, lambda case, level=level, style=style: case["level"] == level and case["style"] == style)
        for stratum in ("context", "none"):
            add(
                f"{base}/{stratum}",
                lambda case, level=level, style=style, stratum=stratum: (
                    case["level"] == level
                    and case["style"] == style
                    and bool(case["context"]) == (stratum == "context")
                ),
            )
    for source in ("c4", "lccc"):
        for stratum in ("context", "none"):
            add(
                f"clause/full/{source}/{stratum}",
                lambda case, source=source, stratum=stratum: (
                    case["level"] == "clause"
                    and case["source"] == source
                    and bool(case["context"]) == (stratum == "context")
                ),
            )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-bin", required=True, help="reference/examples/pool, built with --release")
    parser.add_argument("--model", action="append", default=[], help="name=path of a sentence-model safetensors")
    parser.add_argument("--picks-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.pool, encoding="utf-8") as handle:
        pool = [json.loads(line) for line in handle]
    gold_index = {case["id"]: case["candidates"].index(case["gold"]) for case in pool}
    picks_dir = Path(args.picks_dir)
    picks_dir.mkdir(parents=True, exist_ok=True)

    picks = {"engine": {case["id"]: 0 for case in pool}}
    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        out_path = picks_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as out:
            subprocess.run([args.pool_bin, path, args.pool, "--passes", "1"], stdout=out, check=True)
        with open(out_path, encoding="utf-8") as handle:
            picked = {row["id"]: row["pick"] for row in map(json.loads, handle)}
        if set(picked) != set(gold_index):
            raise SystemExit(f"{name}: picks do not cover the pool")
        picks[name] = picked
        models[name] = {"path": path, "sha256": sha256(path), "picks": str(out_path)}

    report = {"pool": args.pool, "pool_sha256": sha256(args.pool), "cases": len(pool), "models": models, "subsets": {}}
    for key, ids in subsets(pool).items():
        entry = {"cases": len(ids)}
        for name, picked in picks.items():
            hits = sum(picked[case_id] == gold_index[case_id] for case_id in ids)
            entry[name] = {"hits": hits, "rate": round(hits / len(ids), 4), "wilson95": wilson(hits, len(ids))}
        report["subsets"][key] = entry
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    names = list(picks)
    print("| subset | cases | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for key, entry in report["subsets"].items():
        cells = [
            f"{entry[name]['hits']} ({100 * entry[name]['rate']:.1f}%, {100 * entry[name]['wilson95'][0]:.1f}-{100 * entry[name]['wilson95'][1]:.1f})"
            for name in names
        ]
        print(f"| {key} | {entry['cases']} | " + " | ".join(cells) + " |")
    print(f"written {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
