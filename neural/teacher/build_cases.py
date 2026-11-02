"""Sample the distillation cases: clauses typed the way the context eval set types them, from the two training corpora.

The student is trained to pick one candidate from a pool, the way `eval/sentences-ctx-v1/` measures it, so the cases here are cut with that set's rules (`build_set.py` on the `eval/large-context-set` branch; the functions marked "as in build_set.py" below are copies of it, kept identical):

- a clause is a run of 4 to 16 Chinese characters with punctuation, a space or the end of the text on both sides;
- its context is the text before it in the same document or dialogue turn, punctuation kept, at most the last 1024 bytes (what MSIME keeps of the committed text);
- readings come from pypinyin over the whole paragraph or turn, one syllable per character; clauses with a reading full pinyin cannot type are left out;
- every clause gives the clause typed in full pinyin, and a word (or two adjacent words) of 2 to 4 characters inside it, by jieba, typed in full, as initials and mixed. The clause typed as initials or mixed is not written: MSIME composes no sentence from abbreviated input, so those rows never give a pool (`coverage.json` of the eval set counts them).

The text comes from the sources `data/c4.txt` and `data/lccc.txt` were built from, not from those files: `corpus/fetch.py` keeps only the runs of Chinese characters, drops every repeated line across the whole corpus and marks no document boundary, so in those files the line before a clause is not reliably the text before it, and none of its punctuation survives. Reading the raw sources gives the context an input method sees. The same filters as `corpus/fetch.py` apply: traditional documents and turns are skipped (`is_traditional_document`), C4 documents that are not prose are skipped (`is_prose`), and clauses carrying a traditional marker are dropped as `segment` drops them.

- `c4`: documents of the Chinese portion of C4 (ODC-BY), in shard order. Paragraphs are the document's lines, joined with nothing in the context, because Enter is not text an input method commits. A paragraph gives clauses only when it has a comma, full stop, semicolon, colon, exclamation or question mark, which keeps out navigation, menus and titles (the eval set's rule for the government documents).
- `lccc`: dialogues of LCCC-large (MIT). Each turn is a unit; the context is the turn up to the clause, never the other speaker's turns.

Two strata per source, sampled the way the eval set samples its dialogues: units are shuffled and alternate between them; a unit in the `none` stratum gives its first clause if that clause starts the unit (a dialogue turn, or a C4 document), a unit in the `context` stratum gives clauses that do not. A dialogue gives at most three cases, a C4 document at most `C4_PER_DOCUMENT` context cases.

Held out: every dialogue and document belongs to the dev or the train split by a hash of its index, and the dev split is sampled first. A clause is dropped from either split when its gold equals any gold of the eval sets (`eval/sentences-v1.tsv`, `eval/quanpin-words-v1.tsv`, `eval/open-models/pool.jsonl`, `eval/sentences-ctx-v1/sampled.jsonl.gz`), or when its gold or its context shares a stretch of `WINDOW` Chinese characters with any gold or context of those sets. Train clauses are checked the same way against the dev split. Golds are unique across both splits.

usage:
  python neural/teacher/build_cases.py --c4 <c4-zh.tfrecord-00000-of-01024.json.gz> --lccc data/raw/lccc_large.jsonl.gz \\
    --eval-ctx <eval/sentences-ctx-v1/sampled.jsonl.gz> --out-dir <data>/cases

Writes `train.jsonl` and `dev.jsonl` (one typed case per line, the eval set's `sampled.jsonl` fields plus `split`), the same cases as `train.tsv` and `dev.tsv` for MSIME's `convert_eval`, and `build.json` (inputs and their SHA-256, seeds, quotas, what was found and what was dropped). Needs pypinyin and jieba, the versions the eval set was built with (pypinyin 0.55.0, jieba 0.42.1); neither is a dependency of this repository.
"""

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "corpus"))
from fetch import TRADITIONAL, is_prose, is_traditional_document  # noqa: E402

SEED = "teacher-20261030"
MIN_CHARS = 4
MAX_CHARS = 16
WORD_MIN = 2
WORD_MAX = 4
CONTEXT_BYTES = 1024
PER_DIALOGUE = 3
C4_PER_DOCUMENT = 2
# Shared stretches this long count as overlap: the eval set's own overlap check uses 8 (`window8`).
WINDOW = 8
# One unit in this many goes to the dev split.
DEV_SHARE = 25

# Clauses per split, source and stratum: half with context, half without, in both splits. The eval set's clauses without context all come from dialogues, and a C4 document seldom opens with a clause (a title or a navigation line has no punctuation to make its paragraph prose): about one document in nine of shard 0 does, so C4 gives few clauses to the `none` stratum and LCCC the rest.
QUOTAS = {
    "train": {"c4": {"none": 3000, "context": 35000}, "lccc": {"none": 47000, "context": 15000}},
    "dev": {"c4": {"none": 100, "context": 1050}, "lccc": {"none": 1400, "context": 450}},
}
# LCCC-large has twelve million dialogues; this many, picked at random, is more than the quotas need.
LCCC_DIALOGUES = 600000

# --- text, as in build_set.py --------------------------------------------------------------------------

HAN = re.compile(r"[一-鿿]+")
CLAUSE_EDGE = set("，。！？；：、,.!?;:…～~ ")  # noqa: RUF001 (Chinese punctuation)
SPACE = re.compile(r"[\s　\xa0]+")
UNTYPABLE = {"n", "ng", "m", "hm", "hng"}
PROSE_MARK = set("，。；：！？")  # noqa: RUF001 (Chinese punctuation)


def clauses(text):
    """`(start, clause)` for every clause of `text` (build_set.py `clauses`, not strict)."""
    for match in HAN.finditer(text):
        start, end = match.span()
        if start > 0 and text[start - 1] not in CLAUSE_EDGE:
            continue
        if end < len(text) and text[end] not in CLAUSE_EDGE:
            continue
        if MIN_CHARS <= end - start <= MAX_CHARS:
            yield start, match.group()


def tail(text, limit=CONTEXT_BYTES):
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[-limit:].decode("utf-8", errors="ignore")


@lru_cache(maxsize=4096)
def readings(text):
    from pypinyin import Style, lazy_pinyin

    try:
        out = lazy_pinyin(text, style=Style.NORMAL, errors=lambda chars: [None] * len(chars))
    except TypeError:
        # pypinyin 0.55 runs the style conversion over what `errors` returned when the character is
        # a CJK compatibility ideograph (U+F900 block, 年 as U+F98E), and fails on None. Those are not
        # Chinese characters to this set (`HAN`), so a placeholder that converts cleanly stands in for
        # them; text without them takes the path above and reads exactly as build_set.py reads it.
        out = lazy_pinyin(text, style=Style.NORMAL, errors=lambda chars: ["?"] * len(chars))
        out = [None if p == "?" else p for p in out]
    if len(out) != len(text):
        raise SystemExit(f"pinyin does not align with {text!r}")
    return out


def clause_case(source, unit, before, text, start, clause):
    if TRADITIONAL.intersection(clause):
        return None
    pinyin = readings(text)[start : start + len(clause)]
    if any(p is None or not p.isascii() or not p.isalpha() for p in pinyin):
        return None
    if any(p in UNTYPABLE for p in pinyin):
        return None
    return {"source": source, "unit": unit, "gold": clause, "readings": pinyin, "context": tail(before + text[:start])}


def word_window(gold, rng):
    import jieba

    words = jieba.lcut(gold, HMM=True)
    starts = [0]
    for word in words:
        starts.append(starts[-1] + len(word))
    windows = []
    for i in range(len(words)):
        for j in (i + 1, i + 2):
            if j <= len(words) and WORD_MIN <= starts[j] - starts[i] <= WORD_MAX:
                windows.append((starts[i], starts[j] - starts[i]))
    return rng.choice(windows) if windows else None


def typed(pinyin, style, rng):
    if style == "full":
        return "".join(pinyin)
    if style == "initials":
        return "".join(p[0] for p in pinyin)
    while True:
        full = [rng.random() < 0.5 for _ in pinyin]
        if any(full) and not all(full):
            return "".join(p if keep else p[0] for p, keep in zip(pinyin, full, strict=True))


def typed_cases(clause_id, made, rng):
    """build_set.py `typed_cases`, the same random draws in the same order, without the clause typed as initials or mixed."""
    out = []
    shared = {key: made[key] for key in made if key not in ("gold", "readings", "context")}
    for style in ("full", "initials", "mixed"):
        key = typed(made["readings"], style, rng)
        if style != "full":
            continue
        out.append(
            {
                "id": clause_id,
                "level": "clause",
                "style": style,
                "input": key,
                "gold": made["gold"],
                "syllables": len(made["gold"]),
                "readings": made["readings"],
                "context": made["context"],
                **shared,
            }
        )
    window = word_window(made["gold"], rng)
    if window:
        start, length = window
        pinyin = made["readings"][start : start + length]
        context = tail(made["context"] + made["gold"][:start])
        for style in ("full", "initials", "mixed"):
            out.append(
                {
                    "id": f"{clause_id}.w{style[0]}",
                    "level": "word",
                    "style": style,
                    "input": typed(pinyin, style, rng),
                    "gold": made["gold"][start : start + length],
                    "syllables": length,
                    "readings": pinyin,
                    "context": context,
                    "clause": clause_id,
                    **shared,
                }
            )
    return out


# --- overlap with the eval sets ------------------------------------------------------------------------


def windows(text):
    for run in HAN.findall(text):
        for start in range(len(run) - WINDOW + 1):
            yield run[start : start + WINDOW]


class Exclusion:
    """Golds and `WINDOW`-character stretches of texts a sampled clause must not share."""

    def __init__(self):
        self.golds = set()
        self.stretches = set()

    def add(self, gold, context=""):
        self.golds.add(gold)
        self.stretches.update(windows(gold))
        self.stretches.update(windows(context))

    def hit(self, made):
        if made["gold"] in self.golds:
            return "gold"
        if any(stretch in self.stretches for stretch in windows(made["gold"])):
            return "gold_stretch"
        if any(stretch in self.stretches for stretch in windows(made["context"])):
            return "context_stretch"
        return None


def eval_texts(args):
    """`(gold, context)` of every case of the eval sets, and how many each file gave."""
    texts, counts = [], {}

    def tsv(path, gold_field, context_field=None):
        n = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or line.startswith("id\t") or not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                context = fields[context_field] if context_field is not None and len(fields) > context_field else ""
                texts.append((fields[gold_field], context))
                n += 1
        counts[str(path)] = n

    def jsonl(path):
        n = 0
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                texts.append((row["gold"], row.get("context", "")))
                n += 1
        counts[str(path)] = n

    tsv(args.sentences_v1, 2, 5)
    tsv(args.words_v1, 2)
    jsonl(args.open_models_pool)
    jsonl(args.eval_ctx)
    return texts, counts


# --- sources -------------------------------------------------------------------------------------------


def split_of(source, index):
    digest = hashlib.blake2b(f"{SEED}-split-{source}-{index}".encode(), digest_size=8).digest()
    return "dev" if int.from_bytes(digest, "big") % DEV_SHARE == 0 else "train"


def c4_documents(paths):
    """`(index, paragraphs)` of every C4 document that `corpus/fetch.py` would keep."""
    for shard, path in enumerate(paths):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for number, row in enumerate(handle):
                try:
                    text = json.loads(row).get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text or is_traditional_document(text) or not is_prose(text):
                    continue
                paragraphs = [SPACE.sub(" ", line).strip() for line in text.split("\n")]
                yield f"{shard}-{number:06d}", [p for p in paragraphs if p]


def lccc_dialogues(path, rng):
    """`(index, turns)` for a random `LCCC_DIALOGUES` of LCCC-large's dialogues, in file order."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    keep = set(rng.sample(range(total), min(LCCC_DIALOGUES, total)))
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, row in enumerate(handle):
            if number not in keep:
                continue
            row = row.strip()
            if not row or row[0] != "[":
                continue
            try:
                turns = json.loads(row)
            except json.JSONDecodeError:
                continue
            # LCCC ships pre-tokenized with spaces between words; corpus/fetch.py joins them the same way.
            out.append((f"{number:08d}", [SPACE.sub(" ", str(turn).replace(" ", "")).strip() for turn in turns]))
    return out, total


class Sampler:
    """Fills the strata of one split and one source, dropping what the exclusion and the gold de-duplication reject."""

    def __init__(self, split, source, exclusion, taken):
        self.split, self.source = split, source
        self.quotas = QUOTAS[split][source]
        self.out = {"none": [], "context": []}
        self.exclusion, self.taken = exclusion, taken
        self.dropped = {"gold": 0, "gold_stretch": 0, "context_stretch": 0, "duplicate": 0, "untypable": 0}

    def full(self, stratum=None):
        if stratum:
            return len(self.out[stratum]) >= self.quotas[stratum]
        return all(len(self.out[key]) >= self.quotas[key] for key in self.quotas)

    def offer(self, stratum, made):
        """Take `made` into `stratum` unless it is rejected; returns whether it was taken."""
        if made is None:
            self.dropped["untypable"] += 1
            return False
        reason = self.exclusion.hit(made)
        if reason:
            self.dropped[reason] += 1
            return False
        if made["gold"] in self.taken:
            self.dropped["duplicate"] += 1
            return False
        self.taken.add(made["gold"])
        self.out[stratum].append(made)
        return True


def sample_lccc(sampler, dialogues, rng):
    turns = [(f"{index}-{n:02d}", index, text) for index, texts in dialogues for n, text in enumerate(texts)]
    turns = [turn for turn in turns if split_of("lccc", turn[1]) == sampler.split]
    rng.shuffle(turns)
    per_dialogue = {}
    for position, (unit, dialogue, text) in enumerate(turns):
        stratum = "none" if position % 2 == 0 else "context"
        if sampler.full(stratum) or per_dialogue.get(dialogue, 0) >= PER_DIALOGUE:
            continue
        if not text or is_traditional_document(text):
            continue
        options = [(start, clause) for start, clause in clauses(text) if (start == 0) == (stratum == "none")]
        if not options:
            continue
        start, clause = options[0] if stratum == "none" else rng.choice(options)
        if sampler.offer(stratum, clause_case("lccc", unit, "", text, start, clause)):
            per_dialogue[dialogue] = per_dialogue.get(dialogue, 0) + 1
        if sampler.full():
            break


def sample_c4(sampler, documents, rng):
    documents = [document for document in documents if split_of("c4", document[0]) == sampler.split]
    rng.shuffle(documents)
    for position, (unit, paragraphs) in enumerate(documents):
        stratum = "none" if position % 2 == 0 else "context"
        if sampler.full(stratum):
            continue
        if stratum == "none":
            first = paragraphs[0]
            if PROSE_MARK.intersection(first):
                start, clause = next(clauses(first), (None, None))
                if start == 0:
                    sampler.offer("none", clause_case("c4", unit, "", first, 0, clause))
        else:
            options, before = [], ""
            for paragraph in paragraphs:
                if PROSE_MARK.intersection(paragraph):
                    options += [
                        (before, paragraph, start, clause) for start, clause in clauses(paragraph) if before or start
                    ]
                before += paragraph
            rng.shuffle(options)
            taken = 0
            for before, paragraph, start, clause in options:
                if taken == C4_PER_DOCUMENT:
                    break
                taken += sampler.offer("context", clause_case("c4", unit, before, paragraph, start, clause))
        if sampler.full():
            break


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write(rows, jsonl_path, tsv_path, header):
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    # The columns MSIME's `convert_eval` reads: id, input, gold, syllables, tags, context.
    with open(tsv_path, "w", encoding="utf-8") as out:
        out.write(f"# {header}\n")
        for row in rows:
            tags = [row["source"], row["level"], row["style"], "context" if row["context"] else "no-context"]
            fields = [row["id"], row["input"], row["gold"], str(row["syllables"]), ",".join(tags), row["context"]]
            if any("\t" in field or "\n" in field for field in fields):
                raise SystemExit(f"{row['id']}: a tab or a line break in a field")
            out.write("\t".join(fields) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c4", action="append", required=True, help="a c4-zh.tfrecord-*.json.gz shard, in order")
    parser.add_argument("--lccc", required=True, help="lccc_large.jsonl.gz")
    parser.add_argument("--eval-ctx", required=True, help="eval/sentences-ctx-v1/sampled.jsonl.gz")
    parser.add_argument("--sentences-v1", default=str(ROOT / "eval/sentences-v1.tsv"))
    parser.add_argument("--words-v1", default=str(ROOT / "eval/quanpin-words-v1.tsv"))
    parser.add_argument("--open-models-pool", default=str(ROOT / "eval/open-models/pool.jsonl"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import jieba

    jieba.setLogLevel(60)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    texts, eval_counts = eval_texts(args)
    exclusion = Exclusion()
    for gold, context in texts:
        exclusion.add(gold, context)
    print(f"eval sets: {len(exclusion.golds)} golds, {len(exclusion.stretches)} stretches", file=sys.stderr)

    dialogues, lccc_total = lccc_dialogues(args.lccc, random.Random(f"{SEED}-lccc-dialogues"))
    documents = list(c4_documents(args.c4))
    print(
        f"{len(dialogues)} of {lccc_total} dialogues, {len(documents)} c4 documents ({time.time() - started:.0f} s)",
        file=sys.stderr,
    )

    taken = set()
    report = {"seed": SEED, "quotas": QUOTAS, "window": WINDOW, "dev_share": DEV_SHARE, "splits": {}}
    for split in ("dev", "train"):
        samplers = []
        for source in ("c4", "lccc"):
            sampler = Sampler(split, source, exclusion, taken)
            rng = random.Random(f"{SEED}-{split}-{source}")
            if source == "c4":
                sample_c4(sampler, documents, rng)
            else:
                sample_lccc(sampler, dialogues, rng)
            samplers.append(sampler)
            print(
                f"{split}/{source}: none {len(sampler.out['none'])}, context {len(sampler.out['context'])}; dropped {sampler.dropped} ({time.time() - started:.0f} s)",
                file=sys.stderr,
            )
        rows, clauses_by = [], {}
        for sampler in samplers:
            for stratum in ("none", "context"):
                for n, made in enumerate(sampler.out[stratum], 1):
                    clause_id = f"{split[0]}-{sampler.source}-{stratum[0]}-{n:05d}"
                    rows += [
                        {**row, "split": split}
                        for row in typed_cases(clause_id, made, random.Random(f"{SEED}-{clause_id}"))
                    ]
                clauses_by[f"{sampler.source}/{stratum}"] = len(sampler.out[stratum])
        # A seeded shuffle, so that any prefix of a split is a random sample of it.
        random.Random(f"{SEED}-{split}-order").shuffle(rows)
        write(
            rows,
            out_dir / f"{split}.jsonl",
            out_dir / f"{split}.tsv",
            f"chinese-ime-lm neural/teacher {split} cases, written by build_cases.py",
        )
        kinds = {}
        for row in rows:
            key = f"{row['level']}/{row['style']}/{'context' if row['context'] else 'none'}"
            kinds[key] = kinds.get(key, 0) + 1
        report["splits"][split] = {
            "clauses": clauses_by,
            "typed_cases": len(rows),
            "by_kind": dict(sorted(kinds.items())),
            "dropped": {sampler.source: sampler.dropped for sampler in samplers},
        }
        # Train must stay clear of dev the way both stay clear of the eval sets.
        for sampler in samplers:
            for stratum in ("none", "context"):
                for made in sampler.out[stratum]:
                    exclusion.add(made["gold"], made["context"])

    report["inputs"] = {
        "c4": [{"path": path, "sha256": sha256(path)} for path in args.c4],
        "c4_documents_kept": len(documents),
        "lccc": {"path": args.lccc, "sha256": sha256(args.lccc), "dialogues": lccc_total, "read": len(dialogues)},
        "eval": {path: {"cases": n, "sha256": sha256(path)} for path, n in eval_counts.items()},
    }
    import pypinyin

    report["tools"] = {"pypinyin": pypinyin.__version__, "jieba": jieba.__version__, "python": sys.version.split()[0]}
    (out_dir / "build.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
