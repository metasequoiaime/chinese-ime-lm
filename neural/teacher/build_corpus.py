"""Build the 2.0 student's language-model corpus: whole documents with their punctuation, one per line.

The 1.0 corpus (`corpus/fetch.py`) keeps only runs of Chinese characters, so punctuation, digits and Latin letters are cut out and every run becomes its own line; `neural/train.py` then opens every line with BOS. At inference the model reads BOS followed by the tail of the committed text, punctuation and all (`reference/src/lib.rs`, `prefix_tokens`), so 1.0 meets text it never saw in training: every comma is `<unk>`, and BOS was always followed by the start of a clause, never by the middle of a document. The large context eval set shows the cost on clauses: 24.9M picks 87.0% with the context and 87.5% with it stripped, so the context does nothing for it, while Zhiyi goes from 89.5% to 94.6%.

So here a document stays whole. Each output line is one unit the context could come from:

- `c4`: a C4 document (ODC-BY) that `corpus/fetch.py` would keep (not traditional, prose), its paragraphs joined with nothing, because Enter is not text an input method commits (as in `build_cases.py`). A paragraph that is dropped (below) splits the document there, so no line joins text that was not adjacent.
- `lccc`: one turn of an LCCC-large dialogue (MIT), the unit `build_cases.py` takes a context from; the other speaker's turns are not the user's text.

Within a kept unit, a paragraph is dropped when it is a repeat of one already written (C4 pages share templates; LCCC repeats short turns), when it carries a traditional marker (as `segment` drops such runs), or when it shares a stretch of 8 Chinese characters with any gold or context of the eval sets or of the dev pools, the check `build_cases.py` uses. A unit with a gambling word (`lora.py`'s `SPAM`) is dropped whole. Whitespace runs become one space; nothing else is changed, so full-width punctuation stays full-width as MSIME commits it.

C4 is read from shard 2 on: shards 0 and 1 hold the documents the distillation and dev cases were cut from. The two sources are budgeted separately (`--c4-chars`, `--lccc-chars`, counted in characters written).

usage:
  python neural/teacher/build_corpus.py --c4 <shard 2> --c4 <shard 3> ... --lccc data/raw/lccc_large.jsonl.gz \\
    --eval-ctx <eval/sentences-ctx-v1/sampled.jsonl.gz> --dev-pool <data>/pools/dev.jsonl --out <data>/student/corpus
Writes `c4.txt`, `lccc.txt` (one unit per line) and `build.json` (inputs, budgets, what was kept and dropped).
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from build_cases import ROOT, SPACE, TRADITIONAL, Exclusion, eval_texts, is_prose, is_traditional_document, windows
from lora import SPAM

PROGRESS_EVERY = 20_000_000


def digest(text):
    return hashlib.blake2b(text.encode(), digest_size=8).digest()


class Writer:
    """Applies the paragraph rules to units and writes what is left, until the source's budget is spent."""

    def __init__(self, path, budget, exclusion, seen, progress):
        self.out = open(path, "w", encoding="utf-8")
        self.budget, self.exclusion, self.seen, self.progress = budget, exclusion, seen, progress
        self.chars = self.lines = 0
        self.dropped = {"spam_unit": 0, "repeat": 0, "traditional": 0, "eval_stretch": 0}
        self.next_report = PROGRESS_EVERY

    @property
    def full(self):
        return self.chars >= self.budget

    def unit(self, paragraphs):
        if any(word in paragraph for paragraph in paragraphs for word in SPAM):
            self.dropped["spam_unit"] += 1
            return
        run = []
        for paragraph in paragraphs:
            reason = self.reject(paragraph)
            if reason:
                self.dropped[reason] += 1
                self.flush(run)
                run = []
            else:
                run.append(paragraph)
        self.flush(run)

    def reject(self, paragraph):
        key = digest(paragraph)
        if key in self.seen:
            return "repeat"
        self.seen.add(key)
        if TRADITIONAL.intersection(paragraph):
            return "traditional"
        if any(stretch in self.exclusion.stretches for stretch in windows(paragraph)):
            return "eval_stretch"
        return None

    def flush(self, run):
        if not run:
            return
        text = "".join(run)
        self.out.write(text + "\n")
        self.chars += len(text)
        self.lines += 1
        if self.chars >= self.next_report:
            self.next_report += PROGRESS_EVERY
            self.progress(self)

    def close(self):
        self.out.close()


def c4_units(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for row in handle:
                try:
                    text = json.loads(row).get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text or is_traditional_document(text) or not is_prose(text):
                    continue
                paragraphs = [SPACE.sub(" ", line).strip() for line in text.split("\n")]
                yield [p for p in paragraphs if p]


def lccc_units(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row in handle:
            row = row.strip()
            if not row or row[0] != "[":
                continue
            try:
                turns = json.loads(row)
            except json.JSONDecodeError:
                continue
            for turn in turns:
                # LCCC ships pre-tokenized with spaces between words; joining restores the text, as corpus/fetch.py does.
                text = SPACE.sub(" ", str(turn).replace(" ", "")).strip()
                if text and not is_traditional_document(text):
                    yield [text]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c4", action="append", required=True, help="a c4-zh.tfrecord-*.json.gz shard, in order")
    parser.add_argument("--lccc", required=True, help="lccc_large.jsonl.gz")
    parser.add_argument("--eval-ctx", required=True, help="eval/sentences-ctx-v1/sampled.jsonl.gz")
    parser.add_argument("--sentences-v1", default=str(ROOT / "eval/sentences-v1.tsv"))
    parser.add_argument("--words-v1", default=str(ROOT / "eval/quanpin-words-v1.tsv"))
    parser.add_argument("--open-models-pool", default=str(ROOT / "eval/open-models/pool.jsonl"))
    parser.add_argument("--dev-pool", required=True, help="the dev pools, held out from the student as from the teacher")
    parser.add_argument("--c4-chars", type=int, default=300_000_000)
    parser.add_argument("--lccc-chars", type=int, default=300_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", help="a JSON file to keep up to date for the dashboard")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    exclusion = Exclusion()
    texts, eval_counts = eval_texts(args)
    with open(args.dev_pool, encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            texts.append((case["gold"], case["context"]))
    for gold, context in texts:
        exclusion.add(gold, context)
    print(f"{len(texts)} eval and dev cases, {len(exclusion.stretches)} stretches", file=sys.stderr)

    budget = args.c4_chars + args.lccc_chars
    written = {"c4": 0, "lccc": 0}

    def progress(writer, source=None):
        if source:
            written[source] = writer.chars
        total = sum(written.values())
        line = f"{total / 1e6:.0f}M / {budget / 1e6:.0f}M 字"
        print(f"{time.strftime('%T')} {line} ({time.time() - started:.0f} s)", file=sys.stderr)
        if args.progress:
            elapsed = time.time() - started
            eta = elapsed / total * (budget - total) if total else None
            stage = {"status": "running", "progress": total / budget, "detail": line, "eta_s": eta}
            update_progress(args.progress, "corpus", stage)

    seen = set()
    report = {"inputs": {"c4": args.c4, "lccc": args.lccc}, "eval_counts": eval_counts, "budgets": {"c4": args.c4_chars, "lccc": args.lccc_chars}, "sources": {}}
    for source, units, limit in (("c4", c4_units(args.c4), args.c4_chars), ("lccc", lccc_units(args.lccc), args.lccc_chars)):
        writer = Writer(out / f"{source}.txt", limit, exclusion, seen, lambda w, s=source: progress(w, s))
        for paragraphs in units:
            writer.unit(paragraphs)
            if writer.full:
                break
        writer.close()
        written[source] = writer.chars
        report["sources"][source] = {"chars": writer.chars, "lines": writer.lines, "dropped": writer.dropped, "budget_reached": writer.full}
        print(f"{source}: {writer.chars:,} chars in {writer.lines:,} lines, dropped {writer.dropped}", file=sys.stderr)
    write_sources(out, report)
    report["seconds"] = round(time.time() - started)
    (out / "build.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.progress:
        total = sum(written.values())
        update_progress(args.progress, "corpus", {"status": "done", "progress": 1.0, "detail": f"{total / 1e6:.0f}M 字", "eta_s": 0})


SOURCES = {
    "c4": {"name": "the Chinese portion of C4", "license": "ODC-BY", "url": "https://huggingface.co/datasets/allenai/c4", "share_alike": False},
    "lccc": {"name": "LCCC", "license": "MIT", "url": "https://github.com/thu-coai/CDial-GPT", "share_alike": False, "split": "large"},
}


def write_sources(out, report):
    """The licence record `neural/export.py` requires beside every corpus file, in the shape `corpus/fetch.py` writes."""
    for source, record in SOURCES.items():
        made = report["sources"][source]
        entry = {**record, "source": source, "lines": made["lines"], "chars": made["chars"], "builder": "neural/teacher/build_corpus.py"}
        (out / f"{source}.txt.source.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_progress(path, key, stage):
    """Merge one stage into the dashboard's progress file, written atomically."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    data[key] = stage
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
