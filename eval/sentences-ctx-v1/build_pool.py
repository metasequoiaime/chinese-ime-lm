"""Cut the candidate pools of the context set from MSIME's candidate lists, and count what fell outside.

The input is `convert_eval --dump` run on `sampled.tsv` with no model in the resource directory, so
every list is in the engine's own order, recorded after the case's context was committed.

Two rules, one per kind of case:

- Clauses typed in full pinyin: exactly `eval/open-models/build_pool.py`'s rule, the candidates from
  the decoder sources `generated` (8) and `fallback` (9) with as many characters as the gold, in the
  engine's order.
- Words typed in full, as initials or mixed: MSIME answers these from its dictionary, not its
  decoder, so the rule cannot be the decoder sources. The pool is the first nine distinct candidates
  with as many characters as the gold, from any source, in the engine's order. Nine is how many rows
  MSIME's reranker compares (`CONTEST_LIMIT`, one page) and what Qingjian's default tier gives the
  choice model; Zhiyi compares up to twice its page. The length gate needs the gold only for its
  length, which the typed key gives a runtime too: a full syllable or an initial is one character.
  A pool of one candidate is no decision and is left out.

Clauses typed as initials or mixed give no pool: MSIME composes no sentence from abbreviated input
(`coverage.json` counts how often the gold was anywhere in the list at all).

A case whose gold is not in its pool is dropped, as no scorer can reach it; how many were dropped,
per subset, goes to `coverage.json` as a measurement of the engine's candidate generation.

`--stripped` writes the cases that have a context once more with the context taken away and nothing
else changed, same ids, same pools: running a scorer on both files asks what the context did for it.
The pools stay the ones the engine gave with the context committed, so only the scorer's input
changes.

`--overlap` marks each clause with the corpora `check_overlap.py` found it in verbatim (`seen_in`),
as a whole run of the corpus or with the run before it too.

usage:
  python eval/sentences-ctx-v1/build_pool.py --strip eval/sentences-ctx-v1/pool.jsonl --stripped <dir>/pool-stripped.jsonl
  python eval/sentences-ctx-v1/build_pool.py --dump dump.jsonl --sampled <dir>/sampled.jsonl --overlap eval/sentences-ctx-v1/overlap.json \
    --out eval/sentences-ctx-v1/pool.jsonl --stripped <dir>/pool-stripped.jsonl --coverage eval/sentences-ctx-v1/coverage.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "open-models"))
from build_pool import pool as decoder_pool

WORD_POOL = 9


def word_pool(case):
    width = len(case["gold"])
    texts, sources = [], []
    for candidate in case["candidates"]:
        if len(candidate["text"]) == width and candidate["text"] not in texts:
            texts.append(candidate["text"])
            sources.append(candidate["source"])
        if len(texts) == WORD_POOL:
            break
    return texts, sources


SOURCE_NAMES = {0: "database", 1: "user", 8: "generated", 9: "fallback"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump")
    parser.add_argument("--sampled")
    parser.add_argument("--out")
    parser.add_argument("--stripped", required=True)
    parser.add_argument("--overlap")
    parser.add_argument("--coverage")
    parser.add_argument("--strip", help="only write --stripped from this pool.jsonl (no dump needed)")
    args = parser.parse_args()

    if args.strip:
        with open(args.strip, encoding="utf-8") as source, open(args.stripped, "w", encoding="utf-8") as stripped:
            for line in source:
                record = json.loads(line)
                if record["context"]:
                    stripped.write(json.dumps({**record, "context": ""}, ensure_ascii=False) + "\n")
        return
    if not (args.dump and args.sampled and args.out and args.overlap and args.coverage):
        parser.error("--dump, --sampled, --out, --overlap and --coverage are needed unless --strip is given")

    seen_in = defaultdict(list)
    with open(args.overlap, encoding="utf-8") as handle:
        for corpus, found in json.load(handle)["corpora"].items():
            for case_id in sorted(set(found["exact_ids"]) | set(found["document_ids"])):
                seen_in[case_id].append(corpus)

    sampled = {}
    with open(args.sampled, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sampled[row["id"]] = row

    counts = defaultdict(lambda: {"typed": 0, "anywhere": 0, "kept": 0, "single": 0, "engine_top1": 0})
    seen = set()
    with (
        open(args.dump, encoding="utf-8") as source,
        open(args.out, "w", encoding="utf-8") as out,
        open(args.stripped, "w", encoding="utf-8") as stripped,
    ):
        for line in source:
            case = json.loads(line)
            row = sampled[case["id"]]
            seen.add(case["id"])
            if (
                case["gold"] != row["gold"]
                or case["input"] != row["input"]
                or case.get("context", "") != row["context"]
            ):
                raise SystemExit(f"{case['id']}: the dump does not match sampled.jsonl")
            subset = f"{row['level']}/{row['style']}/{'context' if row['context'] else 'none'}"
            count = counts[subset]
            count["typed"] += 1
            count["anywhere"] += any(candidate["text"] == case["gold"] for candidate in case["candidates"])
            if row["level"] == "clause":
                if row["style"] != "full":
                    continue
                texts, sources = decoder_pool(case)
            else:
                texts, raw = word_pool(case)
                sources = [SOURCE_NAMES.get(value, str(value)) for value in raw]
            if case["gold"] not in texts:
                continue
            if len(texts) < 2:
                count["single"] += 1
                continue
            count["kept"] += 1
            count["engine_top1"] += texts[0] == case["gold"]
            record = {
                "id": case["id"],
                "level": row["level"],
                "style": row["style"],
                "source": row["source"],
                "unit": row["unit"],
                "input": case["input"],
                "gold": case["gold"],
                "syllables": row["syllables"],
                "context": row["context"],
                "candidates": texts,
                "sources": sources,
            }
            if row["level"] == "clause":
                record["seen_in"] = seen_in.get(case["id"], [])
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["context"]:
                stripped.write(json.dumps({**record, "context": ""}, ensure_ascii=False) + "\n")
    if seen != set(sampled):
        raise SystemExit(f"the dump has {len(seen)} of the {len(sampled)} sampled cases")

    report = {key: counts[key] for key in sorted(counts)}
    Path(args.coverage).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key, count in report.items():
        print(
            f"{key:28s} typed {count['typed']:5d}  gold anywhere {count['anywhere']:5d}  kept {count['kept']:5d}  single {count['single']:4d}  engine top-1 {count['engine_top1']:5d}"
        )


if __name__ == "__main__":
    main()
