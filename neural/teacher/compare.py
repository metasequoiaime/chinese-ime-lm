"""Compare the teacher's picks with the reference pickers on one pool: hits, Wilson 95% intervals and exact McNemar tests per subset.

A picker is a name and the files of picks it wrote (`score_pool.py`'s output, or any JSON lines with `id` and `pick`): one for the pool and, optionally, one for the cases that have a context scored without it (`name=full.jsonl,stripped.jsonl`). On `eval/sentences-ctx-v1` the reference pickers (engine order, our two models, Zhiyi, Qwen) come from that set's `picks.json` (`--picks-json`), which holds every scorer's pick on every case in the set's case order; on the dev split they are given as files like the teacher's. The engine's order (the first candidate) is always a picker.

Subsets follow the eval set's `summarize.py`: kind of case (clause or word, typing style), with or without a context, the cases with a context scored without it (`/context-stripped`), and for clauses the source (`gov` against `dialogue` on the eval set, `c4` and `lccc` on the dev split) and, when the pool marks it, the clauses not found verbatim in any checked corpus (`unseen`).

The McNemar test is the eval set's: exact, two-sided, p = min(1, 2 P(X <= min(b, c))) for X ~ Binomial(b + c, 1/2), where b cases only the first picker gets right and c only the second. `--pair a:b` names the pairs to test; every pair is tested in every subset both have.

usage:
  python neural/teacher/compare.py --pool <pool.jsonl> [--picks-json <picks.json> --reference ours-25m ...] \\
    --picker zs-9b=<full.jsonl>,<stripped.jsonl> --pair zs-9b:ours-25m --out <results.json> --markdown <results.md>
"""

import argparse
import json
import math
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


def mcnemar(a, b):
    """The eval set's `summarize.py` test."""
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n) if n else 1.0
    return only_a, only_b, p


def read(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def subsets(pool):
    """`key -> [case ids]`, in report order; `/context-stripped` keys hold the cases with a context."""
    out = {}

    def add(key, test):
        ids = [case["id"] for case in pool if test(case)]
        if ids:
            out[key] = ids

    for level, style in [("clause", "full"), ("word", "full"), ("word", "initials"), ("word", "mixed")]:
        base = f"{level}/{style}"

        def mine(case, level=level, style=style):
            return case.get("level", "clause") == level and case.get("style", "full") == style

        add(base, mine)
        add(f"{base}/context", lambda case, mine=mine: mine(case) and case.get("context"))
        add(f"{base}/none", lambda case, mine=mine: mine(case) and not case.get("context"))
        add(f"{base}/context-stripped", lambda case, mine=mine: mine(case) and case.get("context"))
    clause = lambda case: case.get("level", "clause") == "clause"  # noqa: E731
    sources = {case.get("source") for case in pool}
    if "gov" in sources:
        add("clause/full/gov", lambda case: clause(case) and case["source"] == "gov")
        add("clause/full/dialogue", lambda case: clause(case) and case["source"] != "gov")
        add("clause/full/dialogue/context", lambda case: clause(case) and case["source"] != "gov" and case["context"])
        add("clause/full/unseen", lambda case: clause(case) and not case.get("seen_in"))
    for source in ("c4", "lccc"):
        if source in sources:
            for stratum in ("context", "none"):
                add(
                    f"clause/full/{source}/{stratum}",
                    lambda case, source=source, stratum=stratum: (
                        clause(case) and case["source"] == source and bool(case["context"]) == (stratum == "context")
                    ),
                )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--picks-json", help="eval/sentences-ctx-v1/picks.json")
    parser.add_argument("--reference", action="append", default=[], help="a picker id in --picks-json to include")
    parser.add_argument("--picker", action="append", default=[], help="name=full.jsonl[,stripped.jsonl]")
    parser.add_argument("--pair", action="append", default=[], help="a:b, tested in every subset both have")
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown")
    parser.add_argument(
        "--table",
        default="clause/full,clause/full/context,clause/full/none,clause/full/context-stripped,word/full,word/initials,word/mixed",
        help="subsets in the markdown table",
    )
    args = parser.parse_args()

    pool = read(args.pool)
    by_id = {case["id"]: case for case in pool}
    report = subsets(pool)
    pickers = {"engine": ({i: 0 for i in by_id}, None)}

    if args.picks_json:
        saved = json.loads(Path(args.picks_json).read_text(encoding="utf-8"))
        if set(saved["cases"]) != set(by_id):
            raise SystemExit("picks.json is not for this pool")
        for name in args.reference:
            entry = saved["pickers"][name]
            full = dict(zip(saved["cases"], entry["full"], strict=True))
            stripped = (
                dict(zip(saved["stripped_cases"], entry["stripped"], strict=True)) if entry.get("stripped") else None
            )
            pickers[name] = (full, stripped)
    for spec in args.picker:
        name, files = spec.split("=", 1)
        paths = files.split(",")
        full = {row["id"]: row["pick"] for row in read(paths[0])}
        stripped = {row["id"]: row["pick"] for row in read(paths[1])} if len(paths) > 1 else None
        if not set(full) <= set(by_id):
            raise SystemExit(f"{name}: cases that are not in the pool")
        pickers[name] = (full, stripped)

    def outcomes(name, key):
        full, stripped = pickers[name]
        source = stripped if key.endswith("/context-stripped") else full
        ids = report[key]
        if source is None or any(i not in source for i in ids):
            return None
        return [by_id[i]["candidates"][source[i]] == by_id[i]["gold"] for i in ids]

    table = {name: {key: outcomes(name, key) for key in report} for name in pickers}
    results = {
        "pool": args.pool,
        "n": len(pool),
        "subsets": {key: len(ids) for key, ids in report.items()},
        "pickers": {},
        "pairs": {},
    }
    for name in pickers:
        results["pickers"][name] = {
            key: {
                "hits": sum(hits),
                "n": len(hits),
                "rate": round(sum(hits) / len(hits), 4),
                "wilson95": wilson(sum(hits), len(hits)),
            }
            for key, hits in table[name].items()
            if hits is not None
        }
    for pair in args.pair:
        a, b = pair.split(":")
        results["pairs"][pair] = {}
        for key in report:
            if table[a][key] is None or table[b][key] is None:
                continue
            only_a, only_b, p = mcnemar(table[a][key], table[b][key])
            results["pairs"][pair][key] = {
                "n": len(report[key]),
                "hits_a": sum(table[a][key]),
                "hits_b": sum(table[b][key]),
                "only_a": only_a,
                "only_b": only_b,
                "p": p,
            }
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    if args.markdown:
        keys = [key for key in args.table.split(",") if key in report]
        lines = [
            "| Picker | " + " | ".join(f"{key} ({len(report[key])})" for key in keys) + " |",
            "|---" * (len(keys) + 1) + "|",
        ]
        for name in pickers:
            cells = []
            for key in keys:
                entry = results["pickers"][name].get(key)
                cells.append(
                    "-"
                    if entry is None
                    else f"{entry['hits']} ({entry['rate'] * 100:.1f}%, {entry['wilson95'][0] * 100:.1f}-{entry['wilson95'][1] * 100:.1f})"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "| Pair (a:b) | Subset | n | a | b | only a | only b | p |", "|---|---|---|---|---|---|---|---|"]
        for pair, rows in results["pairs"].items():
            for key in keys:
                if key in rows:
                    r = rows[key]
                    lines.append(
                        f"| {pair} | {key} | {r['n']} | {r['hits_a']} | {r['hits_b']} | {r['only_a']} | {r['only_b']} | {r['p']:.2g} |"
                    )
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
