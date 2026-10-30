"""Cut the candidate pool every scorer in the comparison chooses from.

The input is a `convert_eval --dump` from the input method's engine, run with no model in the
resource directory so the order is the engine's own. A case's pool is its candidates from the two
decoder sources, `generated` (8) and `fallback` (9), that have as many characters as the gold, in
the engine's order. That gate needs the gold only for its length, and a runtime has the same fact in
another form: these sources answer the whole key, and every sentence in the set is one character per
syllable. Cases whose gold is not in the pool are dropped, since no scorer can reach them.

usage:
  python eval/open-models/build_pool.py --dump engine-dump.jsonl --set eval/sentences-v1.tsv --out eval/open-models/pool.jsonl
"""

import argparse
import json

DECODER_SOURCES = {8: "generated", 9: "fallback"}


def read_set(path):
    """`id -> (syllables, tags)` from the sentence set, for the fields the dump does not carry."""
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("id\t"):
                continue
            fields = line.split("\t")
            rows[fields[0]] = (int(fields[3]), fields[4].split(",") if len(fields) > 4 and fields[4] else [])
    return rows


def pool(case):
    """The case's decoder candidates of the gold's length, in engine order, first occurrence kept."""
    width = len(case["gold"])
    texts, sources = [], []
    for candidate in case["candidates"]:
        if (
            candidate["source"] in DECODER_SOURCES
            and len(candidate["text"]) == width
            and candidate["text"] not in texts
        ):
            texts.append(candidate["text"])
            sources.append(DECODER_SOURCES[candidate["source"]])
    return texts, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True, help="convert_eval --dump output, engine order (no model loaded)")
    parser.add_argument("--set", required=True, help="the sentence set the dump was recorded from")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_set(args.set)
    kept = dropped = 0
    with open(args.dump, encoding="utf-8") as source, open(args.out, "w", encoding="utf-8") as out:
        for line in source:
            case = json.loads(line)
            texts, sources = pool(case)
            if case["gold"] not in texts:
                dropped += 1
                continue
            syllables, tags = rows[case["id"]]
            record = {
                "id": case["id"],
                "input": case["input"],
                "gold": case["gold"],
                "syllables": syllables,
                "tags": tags,
                "context": case.get("context", ""),
                "candidates": texts,
                "sources": sources,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{kept} cases kept, {dropped} without the gold in the pool")


if __name__ == "__main__":
    main()
