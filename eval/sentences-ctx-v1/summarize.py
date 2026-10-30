"""Turn every scorer's picks on the context set into `results.json`, `results.csv` and `picks.json`.

Each scorer in `models.json` names the files its adapter wrote (one JSON object per case with `id`,
`pick` and `ms`): `picks` for `pool.jsonl` and, for a scorer that reads the context, `stripped` for
`pool-stripped.jsonl` (`build_pool.py --strip`), the cases that have a context with it taken away. A scorer that cannot take some
cases (含章·通变 reads complete pinyin only) leaves them out of its file, and every subset that holds
one of them is reported as not run for it.

Per scorer and subset: hits, Wilson 95% interval, and, only when `--timing-dir` is given, the median and nearest-rank 95th percentile decision time over the subset's cases in that separate timing run (`run_scorers.sh` with TIMING_ONLY=1 on `pool-timing.jsonl`, the 480-case stratified subset `timing_pool.py` draws; the accuracy run's own times are never reported). A timed case must get the same pick as in the accuracy run.
Per subset and every pair of scorers: exact McNemar (two-sided; with `b` cases only the first gets
right and `c` only the second, p = min(1, 2 P(X <= min(b, c))) for X ~ Binomial(b + c, 1/2)), the
discordance (b + c) / n, and the net difference the subset can detect at that discordance.

The detectable difference is `docs/measurements.md`'s power method (alpha 0.05 two-sided, power
0.8), which reproduces that page's table: with discordance d and net difference e, the better scorer
wins a share q = (d + e) / 2d of the discordant cases, and the discordant cases needed are
D = (z_a / 2 + z_b sqrt(q (1 - q)))^2 / (q - 1/2)^2, so n = D / d. Here n and d are measured and e is
solved for.

usage:
  python eval/sentences-ctx-v1/summarize.py --picks-dir <dir> [--timing-dir <dir>] [--out-dir eval/sentences-ctx-v1]
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
Z_A = 1.959963984540054
Z_B = 0.8416212335729143
REFERENCES = ["engine", "ours-4m", "ours-25m"]
PRIMARY = "clause/full"
TA = set("他她它")


def percentile(values, share):
    """Nearest rank, so the figure is a decision that actually happened (as `reference/examples/pool.rs`)."""
    ordered = sorted(values)
    return ordered[min(len(ordered), max(1, math.ceil(share * len(ordered)))) - 1]


def wilson(hits, n):
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    denominator = 1 + Z_A * Z_A / n
    centre = (p + Z_A * Z_A / (2 * n)) / denominator
    half = Z_A * math.sqrt(p * (1 - p) / n + Z_A * Z_A / (4 * n * n)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def needed(discordance, net):
    """Cases needed to detect `net` at `discordance` (the page's method)."""
    q = (discordance + net) / (2 * discordance)
    if q >= 1:
        q = 1 - 1e-12
    pairs = (Z_A / 2 + Z_B * math.sqrt(q * (1 - q))) ** 2 / (q - 0.5) ** 2
    return pairs / discordance


def detectable(n, discordance):
    """The smallest net difference (a fraction) `n` cases detect at `discordance`."""
    if n == 0 or discordance <= 0:
        return None
    low, high = 1e-6, discordance
    if needed(discordance, high * 0.999999) > n:
        return None
    for _ in range(100):
        middle = (low + high) / 2
        if needed(discordance, middle) > n:
            low = middle
        else:
            high = middle
    return high


def mcnemar(a, b):
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n) if n else 1.0
    return only_a, only_b, n, p


def read(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ta_swap(case):
    """The pool holds a candidate that differs from the gold only in 他/她/它: without context the
    pinyin cannot tell them apart."""
    gold = case["gold"]
    for text in case["candidates"]:
        if text != gold and all(x == y or (x in TA and y in TA) for x, y in zip(text, gold, strict=True)):
            return True
    return False


def subsets(pool, stripped_ids):
    """`id -> (label, [case ids])`, in report order."""
    out = {}

    def add(key, label, test):
        ids = [case["id"] for case in pool if test(case)]
        if ids:
            out[key] = (label, ids)

    for level, style in [("clause", "full"), ("word", "full"), ("word", "initials"), ("word", "mixed")]:
        base = f"{level}/{style}"
        mine = lambda case, level=level, style=style: case["level"] == level and case["style"] == style  # noqa: E731
        add(base, f"{level}, {style}", mine)
        add(
            f"{base}/context", f"{level}, {style}, with context", lambda case, mine=mine: mine(case) and case["context"]
        )
        add(f"{base}/none", f"{level}, {style}, no context", lambda case, mine=mine: mine(case) and not case["context"])
        add(
            f"{base}/context-stripped",
            f"{level}, {style}, with context, scored without it",
            lambda case, mine=mine: mine(case) and case["id"] in stripped_ids,
        )
    clause = lambda case: case["level"] == "clause"  # noqa: E731
    add(
        "clause/full/gov",
        "clause, full, government documents (all with context)",
        lambda case: clause(case) and case["source"] == "gov",
    )
    add(
        "clause/full/gov/context-stripped",
        "clause, full, government documents, scored without the context",
        lambda case: clause(case) and case["source"] == "gov" and case["id"] in stripped_ids,
    )
    add("clause/full/dialogue", "clause, full, dialogue", lambda case: clause(case) and case["source"] != "gov")
    add(
        "clause/full/dialogue/context",
        "clause, full, dialogue, with context",
        lambda case: clause(case) and case["source"] != "gov" and case["context"],
    )
    add(
        "clause/full/dialogue/context-stripped",
        "clause, full, dialogue, with context, scored without it",
        lambda case: clause(case) and case["source"] != "gov" and case["id"] in stripped_ids,
    )
    add(
        "clause/full/unseen",
        "clause, full, not found verbatim in any checked corpus",
        lambda case: clause(case) and not case.get("seen_in"),
    )
    add(
        "clause/full/ta",
        "clause, full, pool has a 他/她/它 swap of the gold",
        lambda case: clause(case) and ta_swap(case),
    )
    add(
        "clause/full/ta/none",
        "clause, full, 他/她/它 swap, no context",
        lambda case: clause(case) and ta_swap(case) and not case["context"],
    )
    add(
        "clause/full/ta/context",
        "clause, full, 他/她/它 swap, with context",
        lambda case: clause(case) and ta_swap(case) and case["context"],
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--picks-dir", required=True)
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument(
        "--timing-dir",
        help="a second run on pool-timing.jsonl on a quiet machine (run_scorers.sh with TIMING_ONLY=1); the only source of decision times, and its picks must be the same. Without it every time field is left out and `timing` says pending.",
    )
    args = parser.parse_args()

    manifest = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    pool = read(HERE / "pool.jsonl")
    by_id = {case["id"]: case for case in pool}
    # pool-stripped.jsonl is these cases with the context taken away (build_pool.py --strip).
    stripped_ids = {case["id"] for case in pool if case["context"]}
    picks_dir = Path(args.picks_dir)
    report_subsets = subsets(pool, stripped_ids)

    def load(name, expected):
        rows = read(picks_dir / f"{name}.jsonl")
        ids = [row["id"] for row in rows]
        if not set(ids) <= expected or len(set(ids)) != len(ids):
            raise SystemExit(f"{name}: cases that are not in the pool, or repeated")
        return {row["id"]: row for row in rows}

    # Every picker: a scorer's main run, and for Qwen each start variant too, so that pairs can be tested for all of them.
    pickers = []
    for scorer in manifest["scorers"]:
        if scorer["id"] == "engine":
            pickers.append(
                {"id": "engine", "scorer": scorer, "full": {c["id"]: {"pick": 0} for c in pool}, "stripped": None}
            )
            continue
        variants = scorer.get("start_variants", {"": scorer["picks"]})
        for start, name in variants.items():
            # A scorer not run yet has no file: every subset is reported as not run for it.
            if not (picks_dir / f"{name}.jsonl").exists():
                print(f"{name}: no picks, reported as not run")
                pickers.append(
                    {
                        "id": f"{scorer['id']}@{start}" if start else scorer["id"],
                        "scorer": scorer,
                        "start": start,
                        "full": {},
                        "stripped": None,
                    }
                )
                continue
            full = load(name, set(by_id))
            stripped_name = (
                (scorer["stripped"][start] if start else scorer["stripped"]) if scorer.get("stripped") else None
            )
            stripped = (
                load(stripped_name, stripped_ids)
                if stripped_name and (picks_dir / f"{stripped_name}.jsonl").exists()
                else None
            )
            if args.timing_dir and (Path(args.timing_dir) / f"{name}.jsonl").exists():
                timed = {row["id"]: row for row in read(Path(args.timing_dir) / f"{name}.jsonl")}
                if not set(timed) <= set(full) or any(timed[i]["pick"] != full[i]["pick"] for i in timed):
                    raise SystemExit(f"{name}: the timing run picks differently")
                # Only the timed cases carry a time; the others keep their pick and nothing else.
                full = {
                    i: {**{key: value for key, value in row.items() if key != "ms"}, **({"ms": timed[i]["ms"]} if i in timed else {})}
                    for i, row in full.items()
                }
            else:
                # Times from the accuracy run are not reported: it shares the machine with other work.
                full = {i: {key: value for key, value in row.items() if key != "ms"} for i, row in full.items()}
            if stripped is not None:
                # The timing run covers pool.jsonl only.
                stripped = {i: {key: value for key, value in row.items() if key != "ms"} for i, row in stripped.items()}
            pickers.append(
                {
                    "id": f"{scorer['id']}@{start}" if start else scorer["id"],
                    "scorer": scorer,
                    "start": start,
                    "full": full,
                    "stripped": stripped,
                }
            )

    def outcomes(picker, key):
        """`[hit per case]` for the subset, None if the picker has no pick for one of its cases."""
        _, ids = report_subsets[key]
        source = picker["stripped"] if key.endswith("/context-stripped") else picker["full"]
        if key.endswith("/context-stripped") and source is None:
            # A scorer that reads no context: its picks with the context taken away are its picks.
            source = picker["full"] if picker["scorer"]["inputs"] in ("pinyin", "none") else None
        if source is None or any(case_id not in source for case_id in ids):
            return None, None
        hits = [by_id[i]["candidates"][source[i]["pick"]] == by_id[i]["gold"] for i in ids]
        times = [source[i]["ms"] for i in ids if "ms" in source[i]]
        return hits, times

    table = {}
    for picker in pickers:
        table[picker["id"]] = {key: outcomes(picker, key) for key in report_subsets}

    results_scorers = []
    for scorer in manifest["scorers"]:
        entry = {key: value for key, value in scorer.items() if key not in ("picks", "stripped", "start_variants")}
        main_id = scorer["id"] if "start_variants" not in scorer else f"{scorer['id']}@{scorer['primary_start']}"
        entry["subsets"] = {}
        for key in report_subsets:
            hits, times = table[main_id][key]
            if hits is None:
                entry["subsets"][key] = {"run": False}
                continue
            item = {
                "run": True,
                "hits": sum(hits),
                "n": len(hits),
                "rate": round(sum(hits) / len(hits), 4),
                "wilson95": wilson(sum(hits), len(hits)),
            }
            if times:
                item["timed_n"] = len(times)
                item["decision_ms_p50"] = round(statistics.median(times), 2)
                item["decision_ms_p95"] = round(percentile(times, 0.95), 2)
            if "start_variants" in scorer:
                item["start_variants"] = {}
                for start in scorer["start_variants"]:
                    variant_hits, _ = table[f"{scorer['id']}@{start}"][key]
                    item["start_variants"][start] = None if variant_hits is None else sum(variant_hits)
            entry["subsets"][key] = item
        primary = entry["subsets"][PRIMARY]
        entry.update({key: primary[key] for key in ("hits", "n", "rate", "wilson95") if key in primary})
        for key in ("timed_n", "decision_ms_p50", "decision_ms_p95"):
            if key in primary:
                entry[key] = primary[key]
        if "start_variants" in primary:
            entry["start_variants"] = primary["start_variants"]
        source = next(p for p in pickers if p["id"] == main_id)["full"]
        _, ids = report_subsets[PRIMARY]
        if all(i in source for i in ids):
            entry["picks"] = [source[i]["pick"] for i in ids]
            entry["wrong"] = [
                {"id": i, "picked": by_id[i]["candidates"][source[i]["pick"]], "gold": by_id[i]["gold"]}
                for i in ids
                if by_id[i]["candidates"][source[i]["pick"]] != by_id[i]["gold"]
            ]
        results_scorers.append(entry)

    # Every pair, every subset.
    pairs = {}
    power = {}
    for key, (label, ids) in report_subsets.items():
        rows = []
        for i, a in enumerate(pickers):
            hits_a, _ = table[a["id"]][key]
            if hits_a is None:
                continue
            for b in pickers[i + 1 :]:
                hits_b, _ = table[b["id"]][key]
                if hits_b is None:
                    continue
                only_a, only_b, discordant, p = mcnemar(hits_a, hits_b)
                discordance = discordant / len(ids)
                net = detectable(len(ids), discordance)
                rows.append(
                    {
                        "a": a["id"],
                        "b": b["id"],
                        "a_only": only_a,
                        "b_only": only_b,
                        "discordant": discordant,
                        "discordance": round(discordance, 4),
                        "p": float(f"{p:.3g}"),
                        "detectable_net": None if net is None else round(net, 4),
                    }
                )
        pairs[key] = rows
        # The strongest scorers: the top half of the main runs by hits. Their pairs are the ones a new model is compared in.
        mains = []
        for scorer in manifest["scorers"]:
            main_id = scorer["id"] if "start_variants" not in scorer else f"{scorer['id']}@{scorer['primary_start']}"
            hits, _ = table[main_id][key]
            if hits is not None and scorer["id"] != "engine":
                mains.append((sum(hits), main_id))
        mains.sort(reverse=True)
        strong = {main_id for _, main_id in mains[: max(2, len(mains) // 2)]}
        strong_rows = [row for row in rows if row["a"] in strong and row["b"] in strong]
        if strong_rows:
            median = statistics.median(row["discordance"] for row in strong_rows)
            net = detectable(len(ids), median)
            power[key] = {
                "label": label,
                "n": len(ids),
                "strong": sorted(strong),
                "strong_discordance_median": round(median, 4),
                "strong_discordance_range": [
                    min(row["discordance"] for row in strong_rows),
                    max(row["discordance"] for row in strong_rows),
                ],
                "detectable_net_at_median": None if net is None else round(net, 4),
                "needed_for_3pp": round(needed(median, 0.03)) if median > 0.03 else None,
                "needed_for_5pp": round(needed(median, 0.05)) if median > 0.05 else None,
            }

    for entry in results_scorers:
        main_id = (
            entry["id"]
            if "start_variants" not in next(s for s in manifest["scorers"] if s["id"] == entry["id"])
            else f"{entry['id']}@{entry['primary_start']}"
        )
        entry["mcnemar"] = {}
        for reference in REFERENCES:
            if reference == entry["id"]:
                continue
            for row in pairs[PRIMARY]:
                if {row["a"], row["b"]} == {reference, main_id}:
                    flipped = row["a"] != reference
                    entry["mcnemar"][reference] = {
                        "a_only": row["b_only"] if flipped else row["a_only"],
                        "b_only": row["a_only"] if flipped else row["b_only"],
                        "discordant": row["discordant"],
                        "p": row["p"],
                    }

    # The 60 hand-written sentences of eval/sentences-v1.tsv, re-recorded with the same engine.
    v1_pool = read(HERE / "pool-sentences-v1.jsonl")
    v1 = {"pool": "pool-sentences-v1.jsonl", "n": len(v1_pool), "scorers": {}}
    v1_hits = {}
    for picker in pickers:
        if picker["id"] == "engine":
            picks = [0] * len(v1_pool)
        else:
            scorer = picker["scorer"]
            name = scorer["start_variants"][picker["start"]] if picker.get("start") else scorer["picks"]
            path = picks_dir / f"{name}.v1.jsonl"
            if not path.exists():
                continue
            rows = read(path)
            if [row["id"] for row in rows] != [case["id"] for case in v1_pool]:
                raise SystemExit(f"{name}.v1: not the v1 pool's cases in order")
            picks = [row["pick"] for row in rows]
        hits = [case["candidates"][pick] == case["gold"] for case, pick in zip(v1_pool, picks, strict=True)]
        v1_hits[picker["id"]] = hits
        v1["scorers"][picker["id"]] = {"hits": sum(hits), "wilson95": wilson(sum(hits), len(hits))}
    for picker_id, hits in v1_hits.items():
        if picker_id != "engine":
            only_a, only_b, discordant, p = mcnemar(v1_hits["engine"], hits)
            v1["scorers"][picker_id]["vs_engine"] = {"a_only": only_a, "b_only": only_b, "p": float(f"{p:.3g}")}

    result = {
        "protocol": "Each scorer picks one candidate from the same pool per case; a case counts when the pick is the gold. `subsets` per scorer; `pairs` are exact McNemar tests between every two pickers in every subset (a, b as named); `power` is the net difference each subset can detect at the discordance measured between its strongest scorers.",
        "set": manifest["set"],
        "pool": manifest["pool"],
        "machine": manifest["machine"],
        "timing": manifest["timing"]
        if args.timing_dir
        else "pending: accuracy only; decision times come from a separate timing run on a quiet machine",
        "primary_subset": PRIMARY,
        "subsets": [{"id": key, "label": label, "n": len(ids)} for key, (label, ids) in report_subsets.items()],
        "scorers": results_scorers,
        "power": power,
        "sentences_v1_on_develop": v1,
        "pairs": pairs,
    }
    out = Path(args.out_dir)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "picks.json").write_text(
        json.dumps(
            {
                "cases": [case["id"] for case in pool],
                "stripped_cases": sorted(stripped_ids, key=[case["id"] for case in pool].index),
                "pickers": {
                    picker["id"]: {
                        "full": [picker["full"][c["id"]]["pick"] if c["id"] in picker["full"] else None for c in pool],
                        "stripped": None
                        if picker["stripped"] is None
                        else [
                            picker["stripped"][i]["pick"] if i in picker["stripped"] else None
                            for i in sorted(stripped_ids, key=[case["id"] for case in pool].index)
                        ],
                    }
                    for picker in pickers
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "scorer",
                "label",
                "group",
                "inputs",
                "subset",
                "hits",
                "n",
                "rate",
                "wilson_low",
                "wilson_high",
                "timed_n",
                "decision_ms_p50",
                "decision_ms_p95",
                "weights_bytes",
            ]
        )
        for entry in results_scorers:
            for key, item in entry["subsets"].items():
                if not item["run"]:
                    continue
                writer.writerow(
                    [
                        entry["id"],
                        entry["label"],
                        entry["group"],
                        entry["inputs"],
                        key,
                        item["hits"],
                        item["n"],
                        item["rate"],
                        *item["wilson95"],
                        item.get("timed_n", ""),
                        item.get("decision_ms_p50", ""),
                        item.get("decision_ms_p95", ""),
                        entry.get("weights_bytes", ""),
                    ]
                )

    keys = list(report_subsets)
    print(
        "scorer".ljust(22)
        + "".join(key.replace("clause/", "c/").replace("word/", "w/")[:18].rjust(19) for key in keys[:8])
    )
    for entry in results_scorers:
        cells = []
        for key in keys[:8]:
            item = entry["subsets"][key]
            cells.append((f"{item['hits']}/{item['n']}" if item["run"] else "-").rjust(19))
        print(entry["id"].ljust(22) + "".join(cells))
    for key, item in power.items():
        print(
            f"power {key}: n={item['n']} discordance median {item['strong_discordance_median']:.3f} -> detectable {item['detectable_net_at_median']}"
        )


if __name__ == "__main__":
    main()
