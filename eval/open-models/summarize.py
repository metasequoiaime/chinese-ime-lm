"""Turn every scorer's per-case picks into the comparison's numbers: `results.json` and `results.csv`.

Each scorer in `models.json` names a picks file (one JSON object per case, in pool order, with `pick`
and `ms`) and the files whose `ms` give its latency. For each scorer this reports the hits with a
Wilson 95% interval, the median decision time of each timing run, and exact McNemar tests against
the engine order and this project's two models. A scorer's timing runs must agree on every pick: a
run that picks differently is a different measurement, not a second sample of the same one.

McNemar is exact and two-sided: with `b` cases only the first scorer gets right and `c` only the
second does, p = min(1, 2 * P(X <= min(b, c))) for X ~ Binomial(b + c, 1/2).

usage:
  python eval/open-models/summarize.py --picks-dir <dir> [--out-dir eval/open-models]
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCES = ["engine", "ours-4m", "ours-25m"]
Z95 = 1.959963984540054


def wilson(hits, n):
    p = hits / n
    denominator = 1 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2 * n)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a, b):
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n) if n else 1.0
    return {"a_only": only_a, "b_only": only_b, "discordant": n, "p": round(p, 6)}


def read(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--picks-dir", required=True, help="where the scorers wrote <name>.jsonl")
    parser.add_argument("--out-dir", default=str(HERE))
    args = parser.parse_args()

    manifest = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    pool = read(HERE / manifest["pool"]["file"])
    picks_dir = Path(args.picks_dir)

    def load(name):
        rows = read(picks_dir / f"{name}.jsonl")
        if [row["id"] for row in rows] != [case["id"] for case in pool]:
            raise SystemExit(f"{name}: cases are not the pool's, in the pool's order")
        return rows

    def hits_of(picks):
        return [case["candidates"][pick] == case["gold"] for case, pick in zip(pool, picks, strict=True)]

    scorers = []
    for scorer in manifest["scorers"]:
        entry = dict(scorer)
        if scorer["id"] == "engine":
            picks, timings = [0] * len(pool), []
        else:
            picks = [row["pick"] for row in load(scorer["picks"])]
            timings = []
            for name in scorer["timing"]:
                rows = load(name)
                if scorer["group"] != "general" and [row["pick"] for row in rows] != picks:
                    raise SystemExit(f"{name}: picks differ from {scorer['picks']}")
                timings.append({"run": name, "decision_ms_p50": round(statistics.median(row["ms"] for row in rows), 2)})
            if "cpu_round" in scorer:
                rows = load(scorer["cpu_round"])
                if [row["pick"] for row in rows] != picks:
                    raise SystemExit(f"{scorer['cpu_round']}: picks differ on the CPU")
                entry["cpu_decision_ms_p50"] = round(statistics.median(row["ms"] for row in rows), 2)
        hits = hits_of(picks)
        low, high = wilson(sum(hits), len(pool))
        entry.update(
            {
                "hits": sum(hits),
                "n": len(pool),
                "rate": round(sum(hits) / len(pool), 4),
                "wilson95": [round(low, 4), round(high, 4)],
                "picks": picks,
                "wrong": [
                    {"id": case["id"], "picked": case["candidates"][pick], "gold": case["gold"]}
                    for case, pick, hit in zip(pool, picks, hits, strict=True)
                    if not hit
                ],
            }
        )
        if timings:
            values = [t["decision_ms_p50"] for t in timings]
            entry["decision_ms_p50_runs"] = timings
            entry["decision_ms_p50_range"] = [min(values), max(values)]
        if "start_variants" in scorer:
            entry["start_variants"] = {
                start: sum(hits_of([row["pick"] for row in load(name)]))
                for start, name in scorer["start_variants"].items()
            }
        entry["_hits"] = hits
        scorers.append(entry)

    by_id = {entry["id"]: entry for entry in scorers}
    for entry in scorers:
        entry["mcnemar"] = {
            reference: mcnemar(by_id[reference]["_hits"], entry["_hits"])
            for reference in REFERENCES
            if reference != entry["id"]
        }
    for entry in scorers:
        del entry["_hits"]

    result = {
        "protocol": "Each scorer picks one candidate from the same pool per case; a case counts when the pick is the gold. McNemar entries compare the named reference (a) with this scorer (b).",
        "pool": manifest["pool"],
        "machine": manifest["machine"],
        "scorers": scorers,
    }
    out = Path(args.out_dir)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        header = ["id", "label", "group", "inputs", "hits", "n", "rate", "wilson_low", "wilson_high"]
        header += ["decision_ms_p50_min", "decision_ms_p50_max", "weights_bytes"]
        for reference in REFERENCES:
            header += [f"vs_{reference}_ref_only", f"vs_{reference}_this_only", f"vs_{reference}_p"]
        writer.writerow(header)
        for entry in scorers:
            row = [
                entry["id"],
                entry["label"],
                entry["group"],
                entry["inputs"],
                entry["hits"],
                entry["n"],
                entry["rate"],
            ]
            row += entry["wilson95"]
            row += entry.get("decision_ms_p50_range", ["", ""])
            row.append(entry.get("weights_bytes", ""))
            for reference in REFERENCES:
                test = entry["mcnemar"].get(reference)
                row += [test["a_only"], test["b_only"], test["p"]] if test else ["", "", ""]
            writer.writerow(row)

    for entry in scorers:
        span = entry.get("decision_ms_p50_range")
        timing = f"{span[0]:.1f}-{span[1]:.1f} ms" if span else "-"
        tests = "  ".join(f"vs {ref} {t['a_only']}:{t['b_only']} p={t['p']:.3f}" for ref, t in entry["mcnemar"].items())
        print(
            f"{entry['id']:20s} {entry['hits']}/{entry['n']} {entry['rate']:.3f} {entry['wilson95']} {timing:>14s}  {tests}"
        )


if __name__ == "__main__":
    main()
