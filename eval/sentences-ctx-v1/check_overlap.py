"""Check the sampled clauses against corpora the compared scorers were trained on.

Every corpus is read as runs of Chinese characters, split at every other character, which is how
`corpus/fetch.py` writes this project's training files; Tatoeba and the zhwiki dump are cut the
same way here, and Tatoeba's traditional sentences are converted to simplified first. Three checks
per clause typed in full:

- `exact`: the clause is a whole run of the corpus, the same text between the same punctuation.
- `window8`: some 8-character stretch of the clause occurs anywhere in the corpus (MSIME ran this
  check for its typing sets). Clauses shorter than 8 characters cannot hit.
- `document`: the clause is a run of the corpus and the run before it is the clause's preceding run
  in its own text. Two adjacent runs of the source text in the same order is what it looks like
  when the source document itself is in the corpus, rather than a common phrase.

For the zhwiki dump the clause is also looked for in traditional characters (OpenCC s2t), because
zhwiki keeps each article in the variant its editors used.

usage:
  python eval/sentences-ctx-v1/check_overlap.py --sampled eval/sentences-ctx-v1/sampled.jsonl \
    --lines c4=<chinese-ime-lm>/data/c4.txt --lines lccc=<chinese-ime-lm>/data/lccc.txt \
    --tatoeba <cmn_sentences_detailed.tsv.bz2> --wiki <zhwiki-20260901-pages-articles1.xml-p1p187712.bz2> \
    --out eval/sentences-ctx-v1/overlap.json

Needs opencc-python-reimplemented for the traditional forms.
"""

import argparse
import bz2
import json
import re
import sys
import time

RUN = re.compile(r"[一-鿿]+")
WINDOW = 8


def load(path):
    clauses = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["level"] != "clause" or row["style"] != "full":
                continue
            # The run right before the clause: a clause always follows punctuation or a space, so
            # the context's last run is the stretch of text before that edge.
            runs = RUN.findall(row["context"])
            previous = runs[-1] if runs else ""
            clauses.append({"id": row["id"], "source": row["source"], "gold": row["gold"], "previous": previous})
    return clauses


class Probe:
    def __init__(self, clauses, variants):
        # gold text (or a variant of it) -> clause ids
        self.exact = {}
        self.windows = {}
        self.pairs = {}
        for clause in clauses:
            for gold, previous in variants(clause):
                self.exact.setdefault(gold, set()).add(clause["id"])
                for start in range(len(gold) - WINDOW + 1):
                    self.windows.setdefault(gold[start : start + WINDOW], set()).add(clause["id"])
                if len(previous) >= 2:
                    self.pairs.setdefault((previous, gold), set()).add(clause["id"])
        self.hit = {"exact": set(), "window8": set(), "document": set()}
        self.runs = 0
        self.characters = 0

    def feed(self, runs):
        """Runs of one document, in order."""
        previous = None
        exact, windows, pairs = self.exact, self.windows, self.pairs
        for run in runs:
            self.runs += 1
            self.characters += len(run)
            ids = exact.get(run)
            if ids:
                self.hit["exact"] |= ids
                if previous is not None and (previous, run) in pairs:
                    self.hit["document"] |= pairs[(previous, run)]
            for start in range(len(run) - WINDOW + 1):
                ids = windows.get(run[start : start + WINDOW])
                if ids:
                    self.hit["window8"] |= ids
            previous = run


def line_runs(path):
    """A training file of `corpus/fetch.py`: one run per line, in document order."""
    with open(path, encoding="utf-8") as handle:
        batch = []
        for line in handle:
            batch.append(line.rstrip("\n"))
            if len(batch) == 100000:
                yield batch
                batch = []
        if batch:
            yield batch


def tatoeba_runs(path, t2s):
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "cmn":
                yield RUN.findall(t2s.convert(parts[2]))


def wiki_runs(path):
    """The `<text>` of every page of a pages-articles dump, cut into runs; markup splits runs too."""
    text_open = re.compile(r"<text[^>]*>")
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        inside = False
        page = []
        for line in handle:
            if not inside:
                match = text_open.search(line)
                if not match:
                    continue
                inside = True
                line = line[match.end() :]
            end = line.find("</text>")
            if end >= 0:
                page.append(line[:end])
                yield RUN.findall("\n".join(page))
                page = []
                inside = False
            else:
                page.append(line)


def summarize(probe, clauses):
    by_source = {}
    for clause in clauses:
        entry = by_source.setdefault(
            clause["source"], {"clauses": 0, "exact": 0, "window8": 0, "document": 0, "long": 0, "long_exact": 0}
        )
        entry["clauses"] += 1
        long = len(clause["gold"]) >= WINDOW
        entry["long"] += long
        for check in ("exact", "window8", "document"):
            entry[check] += clause["id"] in probe.hit[check]
        entry["long_exact"] += long and clause["id"] in probe.hit["exact"]
    return {
        "runs": probe.runs,
        "characters": probe.characters,
        "by_source": by_source,
        "document_ids": sorted(probe.hit["document"]),
        "exact_ids": sorted(probe.hit["exact"]),
        "window8_ids": sorted(probe.hit["window8"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sampled", required=True)
    parser.add_argument("--lines", action="append", default=[], help="name=path of a corpus/fetch.py training file")
    parser.add_argument("--tatoeba")
    parser.add_argument("--wiki")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from opencc import OpenCC

    s2t, t2s = OpenCC("s2t"), OpenCC("t2s")
    clauses = load(args.sampled)

    def plain(clause):
        return [(clause["gold"], clause["previous"])]

    def both(clause):
        return [*plain(clause), (s2t.convert(clause["gold"]), s2t.convert(clause["previous"]))]

    report = {"clauses": len(clauses), "window": WINDOW, "corpora": {}}
    for spec in args.lines:
        name, path = spec.split("=", 1)
        started = time.time()
        probe = Probe(clauses, plain)
        for batch in line_runs(path):
            probe.feed(batch)
        report["corpora"][name] = {"path": path, **summarize(probe, clauses)}
        print(f"{name}: {time.time() - started:.0f} s", file=sys.stderr)
    if args.tatoeba:
        probe = Probe(clauses, plain)
        for runs in tatoeba_runs(args.tatoeba, t2s):
            probe.feed(runs)
        report["corpora"]["tatoeba"] = {"path": args.tatoeba, **summarize(probe, clauses)}
    if args.wiki:
        started = time.time()
        probe = Probe(clauses, both)
        for runs in wiki_runs(args.wiki):
            probe.feed(runs)
        report["corpora"]["zhwiki"] = {"path": args.wiki, **summarize(probe, clauses)}
        print(f"zhwiki: {time.time() - started:.0f} s", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")
    for name, corpus in report["corpora"].items():
        print(name, json.dumps(corpus["by_source"], ensure_ascii=False))


if __name__ == "__main__":
    main()
