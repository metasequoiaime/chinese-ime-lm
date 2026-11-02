"""Build the context sentence set: fetch the sources, cut them into typed units, sample.

A clause case is one stretch someone types and commits in one go: a run of Chinese characters that
the text itself delimits with punctuation (or a space) on both sides, 4 to 16 characters long. Its
context is the text written before it in the same document, as an input method sees it: the text
before the caret. The gold is the clause as its author wrote it.

Every sampled clause also gives one word case: a word, or two adjacent words, of 2 to 4 characters
inside the clause (jieba's segmentation), with the text before it as context. Word cases are typed
three ways: in full pinyin, as initials only (the first letter of every syllable, 中国人 `zgr`), and
mixed (some syllables in full, the others as their first letter, 我是一个 `wsyige`). The clause
itself is typed the same three ways, but only the full-pinyin clause makes a pool: MSIME composes no
sentences from abbreviated input, and those rows are kept only to count that (README.md).

Four sources, none of them in the training data of this project's models (Chinese C4 and LCCC), of
Zhiyi's (SkyPile, LCCC, C4, SODA, Tatoeba, FineWeb, FineWeb-2) or in the corpus MSIME's n-gram tables
come from (the zhwiki dump of 2026-09-01), as far as can be known; README.md has the licences and the
overlap checks:

- `gov`: documents of the State Council, its General Office and the ministries in the policy library
  of www.gov.cn, published after 2026-09-01, with their PDF attachments. Formal written prose. Not
  protected by copyright under Article 5 of the Copyright Law of the PRC. Context is the document up
  to the clause; paragraphs are joined with nothing in between, because the Enter key is not text an
  input method commits.
- `kdconv`, `crosswoz`, `risawoz`: the test splits of three crowd-sourced dialogue sets (KdConv and
  CrossWOZ, Apache-2.0; RiSAWOZ, MIT): chat about films, music and travel, and task dialogues about
  hotels, restaurants, attractions and more, typed by the crowd workers. Context is the turn up to
  the clause. The other speaker's turns are not context: they are on the screen, but they are not
  text the user committed, which is all an input method sees.

usage:
  python eval/sentences-ctx-v1/build_set.py fetch --cache <dir>
  python eval/sentences-ctx-v1/build_set.py build --cache <dir> --out eval/sentences-ctx-v1/sampled.jsonl --tsv <dir>/sampled.tsv

`fetch` needs `pdftotext` (poppler) only later, at `build`. `build` needs pypinyin (MIT), jieba (MIT)
and pdftotext; none is a dependency of this repository.
"""

import argparse
import hashlib
import html
import io
import json
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

USER_AGENT = "chinese-ime-lm eval set builder (https://github.com/metasequoiaime/chinese-ime-lm)"
SEED = 20261030
MIN_CHARS = 4
MAX_CHARS = 16
WORD_MIN = 2
WORD_MAX = 4
# Bytes of context kept, from the end: MSIME keeps the last 1024 bytes of committed text, and no
# scorer in the comparison reads further back than that.
CONTEXT_BYTES = 1024

# --- sources ---------------------------------------------------------------------------------------

GOV_LIST = (
    "https://sousuo.www.gov.cn/search-gov/data?t={table}&q=&timetype=timeqb&mintime=&maxtime="
    "&sort=pubtime&sortType=1&searchfield=title&puborg=&pcodeYear=&pcodeNum=&filetype=&p={page}&n=50"
    "&inpro=&bmfl=&dup=&orpro=&type={kind}"
)
GOV_TABLES = [("zhengcelibrary_gw", "gwyzcwjk"), ("zhengcelibrary_bm", "bmwj")]
# The zhwiki dump MSIME's n-gram tables are built from is 20260901, so nothing published after that
# day can be in it, nor in any of the web corpora the other scorers were trained on.
GOV_PUBLISHED_AFTER = "2026.09.01"
GOV_PER_DOCUMENT = 20

DIALOGUES = {
    "kdconv": {
        "commit": "653db76432de09a004ba708a68f8bbd5500e6bec",
        "files": {
            f"{domain}-test.json": f"https://raw.githubusercontent.com/thu-coai/KdConv/{{commit}}/data/{domain}/test.json"
            for domain in ["film", "music", "travel"]
        },
        "license": "Apache-2.0",
        "quotas": {"none": 300, "context": 300},
    },
    "crosswoz": {
        "commit": "df82c9fdff91b9b130f2d6b89110d3870ba6260e",
        "files": {
            "test.json.zip": "https://raw.githubusercontent.com/thu-coai/CrossWOZ/{commit}/data/crosswoz/test.json.zip"
        },
        "license": "Apache-2.0",
        "quotas": {"none": 250, "context": 220},
    },
    "risawoz": {
        "commit": "1e2d3be2b25a0540519db1664a172d85ddd62b29",
        "files": {
            "source-data.zip": "https://raw.githubusercontent.com/terryqj0107/RiSAWOZ/{commit}/RiSAWOZ-data/source-data.zip"
        },
        "license": "MIT",
        "quotas": {"none": 250, "context": 220},
    },
}
PER_DIALOGUE = 3


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(cache):
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved": time.strftime("%Y-%m-%d"), "gov": [], "dialogues": {}}

    documents = []
    for table, kind in GOV_TABLES:
        page = 1
        while True:
            listing = json.loads(get(GOV_LIST.format(table=table, kind=kind, page=page)))
            rows = listing["searchVO"]["listVO"] or []
            fresh = [row for row in rows if row["pubtimeStr"] > GOV_PUBLISHED_AFTER]
            documents += fresh
            if len(fresh) < len(rows) or not rows:
                break
            page += 1
    (cache / "gov").mkdir(exist_ok=True)
    for row in sorted(documents, key=lambda r: r["url"]):
        name = row["url"].rsplit("/", 1)[1].removesuffix(".htm")
        path = cache / "gov" / f"{name}.htm"
        if not path.exists():
            path.write_bytes(get(row["url"]))
            time.sleep(1)
        # Many ministry notices carry the plan or the measures they issue as a PDF attachment.
        attachments = []
        for link in dict.fromkeys(ATTACHMENT.findall(path.read_bytes().decode("utf-8", errors="replace"))):
            url = urllib.parse.urljoin(row["url"], link)
            target = cache / "gov" / f"{name}-{url.rsplit('/', 1)[1]}"
            if not target.exists():
                target.write_bytes(get(url))
                time.sleep(1)
            attachments.append({"url": url, "file": target.name, "sha256": sha256(target.read_bytes())})
        manifest["gov"].append(
            {
                "doc": name,
                "url": row["url"],
                "title": row["title"],
                "code": row["pcode"],
                "issuer": row["puborg"],
                "published": row["pubtimeStr"],
                "sha256": sha256(path.read_bytes()),
                "attachments": attachments,
            }
        )

    for source, spec in DIALOGUES.items():
        files = {}
        for name, template in spec["files"].items():
            url = template.format(commit=spec["commit"])
            path = cache / f"{source}-{name}"
            if not path.exists():
                path.write_bytes(get(url))
            files[name] = {"url": url, "sha256": sha256(path.read_bytes())}
        manifest["dialogues"][source] = {"commit": spec["commit"], "license": spec["license"], "files": files}

    (cache / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(manifest['gov'])} gov documents and {len(DIALOGUES)} dialogue sets in {cache}", file=sys.stderr)


# --- text --------------------------------------------------------------------------------------------

HAN = re.compile(r"[一-鿿]+")
# What may stand on either side of a clause. Anything else next to a run of characters (a digit, a
# Latin letter, a quotation mark, a bracket) means the run is part of something larger that is not
# typed as pinyin, so the run is not a clause. A space counts as an edge: in these documents it
# separates a label from its text (第六条 国家鼓励).
CLAUSE_EDGE = set("，。！？；：、,.!?;:…～~ ")  # noqa: RUF001 (Chinese punctuation)
SPACE = re.compile(r"[\s　\xa0]+")
# Readings standard full pinyin has no syllable for (interjections such as 嗯 n, 呣 m).
UNTYPABLE = {"n", "ng", "m", "hm", "hng"}
ATTACHMENT = re.compile(r'href="([^"]+\.pdf)"')
# A gov paragraph gives clauses only when it is prose, which in these documents always has a comma,
# a full stop or a semicolon somewhere. That keeps out headings, table cells and the lines of a
# PDF's tables, which are not text anyone types as a sentence.
PROSE_MARK = set("，。；：！？")  # noqa: RUF001 (Chinese punctuation)
DIV = re.compile(r"<(/?)div\b[^>]*>")
PARAGRAPH = re.compile(r"(?s)<p[^>]*>(.*?)</p>")
TAG = re.compile(r"<[^>]+>")


def clauses(text, strict=False):
    """`(start, clause)` for every clause of `text`: a run of characters with a clause edge or the
    end of the text on both sides, of a length a user types in one go.

    `strict` is for text out of a PDF, whose paragraph ends and spaces are where pdftotext guessed a
    line broke or a column began, and may fall in the middle of a word (深化电信普 / 遍服务): there
    only punctuation counts as an edge.
    """
    edge = CLAUSE_EDGE - {" "} if strict else CLAUSE_EDGE
    for match in HAN.finditer(text):
        start, end = match.span()
        if (start == 0 and strict) or (start > 0 and text[start - 1] not in edge):
            continue
        if (end == len(text) and strict) or (end < len(text) and text[end] not in edge):
            continue
        if MIN_CHARS <= end - start <= MAX_CHARS:
            yield start, match.group()


def tail(text, limit=CONTEXT_BYTES):
    """The last `limit` bytes of `text`, cut on a character boundary."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[-limit:].decode("utf-8", errors="ignore")


@lru_cache(maxsize=4096)
def readings(text):
    """pypinyin's reading of every character of `text`, read as a whole so that the words around a
    polyphone decide it (还钱 huan, 还是 hai); None for characters that are not Chinese."""
    from pypinyin import Style, lazy_pinyin

    out = lazy_pinyin(text, style=Style.NORMAL, errors=lambda chars: [None] * len(chars))
    if len(out) != len(text):
        raise SystemExit(f"pinyin does not align with {text!r}")
    return out


def gov_paragraphs(raw):
    page = raw.decode("utf-8", errors="replace")
    start = page.find('id="UCAP-CONTENT"')
    if start < 0:
        return []
    # The content is the UCAP-CONTENT div; it ends where its own </div> closes it, not at the
    # page's footer, which some pages put in the same table cell.
    body = page[start:]
    depth = 0
    for tag in DIV.finditer(body):
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            body = body[: tag.start()]
            break
    out = []
    for paragraph in PARAGRAPH.findall(body):
        text = SPACE.sub(" ", html.unescape(TAG.sub("", paragraph))).strip()
        if text:
            out.append(text)
    return out


def spaced(line):
    """A PDF line set letter-spaced (推 动 田 管) with the spaces taken out; other lines unchanged."""
    spaces = line.count(" ")
    characters = sum(len(run) for run in HAN.findall(line))
    if spaces >= 3 and spaces * 2 >= characters:
        return line.replace(" ", "")
    return line


def pdf_paragraphs(path):
    """The paragraphs of a PDF attachment, from `pdftotext` (poppler).

    pdftotext gives the lines as laid out on the page. A Chinese paragraph wraps without hyphens or
    spaces, so the lines of a paragraph are joined with nothing; a line ends its paragraph when it
    is visibly shorter than a full line or ends a sentence, which is how the page shows it too. A
    paragraph that is really a table row or a heading has no comma or full stop and gives no
    clauses (`PROSE_MARK`).
    """
    text = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", "-nopgbrk", str(path), "-"], capture_output=True, check=True, text=True
    ).stdout
    lines = [spaced(SPACE.sub(" ", line).strip()) for line in text.splitlines()]
    lengths = sorted(len(line) for line in lines if len(line) >= 10)
    if not lengths:
        return []
    width = lengths[len(lengths) * 9 // 10]
    out, current = [], ""
    for line in lines:
        # Blank lines and page numbers (— 12 —) are where a page ends, not where a paragraph does:
        # a sentence runs on from the foot of one page to the head of the next.
        if not HAN.search(line) and len(line) <= 8:
            continue
        current += line
        if len(line) < width - 2 or line[-1] in "。！？：；":  # noqa: RUF001 (Chinese punctuation)
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


# --- clauses per source ------------------------------------------------------------------------------


def clause_case(source, unit, before, text, start, clause, extra):
    """The clause at `start` of `text`, with `before` and the part of `text` ahead of it as context.

    The readings come from `text` alone (a paragraph, a turn), which is enough for pypinyin to see
    the words around the clause.
    """
    pinyin = readings(text)[start : start + len(clause)]
    if any(p is None or not p.isascii() or not p.isalpha() for p in pinyin):
        return None
    if any(p in UNTYPABLE for p in pinyin):
        return None
    return {
        "source": source,
        "unit": unit,
        "gold": clause,
        "readings": pinyin,
        "context": tail(before + text[:start]),
        **extra,
    }


def gov_clauses(cache, manifest, rng):
    """Up to `GOV_PER_DOCUMENT` clauses from each document, none already taken from an earlier one:
    the documents share a good deal of boilerplate (为贯彻落实党中央), and a clause taken twice
    would only be dropped later as a duplicate."""
    out = []
    taken = set()
    for document in manifest["gov"]:
        paragraphs = [(text, False) for text in gov_paragraphs((cache / "gov" / f"{document['doc']}.htm").read_bytes())]
        for attachment in document["attachments"]:
            paragraphs += [(text, True) for text in pdf_paragraphs(cache / "gov" / attachment["file"])]
        before = ""
        found = []
        for paragraph, from_pdf in paragraphs:
            if PROSE_MARK.intersection(paragraph):
                for start, clause in clauses(paragraph, strict=from_pdf):
                    made = clause_case("gov", document["doc"], before, paragraph, start, clause, {})
                    if made:
                        found.append(made)
            before += paragraph
        rng.shuffle(found)
        fresh = []
        for made in found:
            if made["gold"] not in taken and len(fresh) < GOV_PER_DOCUMENT:
                taken.add(made["gold"])
                fresh.append(made)
        out += fresh
    return out


def stratified(source, turns, rng, quotas):
    """Sample the two strata of a dialogue set.

    Most turns are one clause, so a clause picked at random from a random turn nearly always starts
    the turn and has no context. The two strata are therefore sampled on their own: the shuffled
    turns alternate between them, a turn in the `none` stratum gives its first clause if that clause
    starts the turn, and a turn in the `context` stratum gives one of its other clauses at random.
    Each stratum is a random sample of the turns that can give one, no turn gives two cases, and no
    dialogue gives more than `PER_DIALOGUE`, so that one long chat does not fill a stratum.
    """
    out = {"none": [], "context": []}
    per_dialogue = {}
    for position, (unit, dialogue, text) in enumerate(turns):
        stratum = "none" if position % 2 == 0 else "context"
        if len(out[stratum]) >= quotas[stratum] or per_dialogue.get(dialogue, 0) >= PER_DIALOGUE:
            continue
        options = [(start, clause) for start, clause in clauses(text) if (start == 0) == (stratum == "none")]
        if not options:
            continue
        start, clause = options[0] if stratum == "none" else rng.choice(options)
        made = clause_case(source, unit, "", text, start, clause, {})
        if made:
            out[stratum].append(made)
            per_dialogue[dialogue] = per_dialogue.get(dialogue, 0) + 1
        if all(len(out[key]) >= quotas[key] for key in quotas):
            break
    return out["none"] + out["context"]


def dialogue_turns(cache, source):
    """`(unit, dialogue, text)` for every turn of the source's test split, both speakers."""
    clean = lambda text: SPACE.sub(" ", text).strip()  # noqa: E731
    turns = []
    if source == "kdconv":
        for domain in ["film", "music", "travel"]:
            dialogues = json.loads((cache / f"kdconv-{domain}-test.json").read_text(encoding="utf-8"))
            for number, dialogue in enumerate(dialogues):
                for index, message in enumerate(dialogue["messages"]):
                    turns.append(
                        (f"{domain}-{number:03d}-{index:02d}", f"{domain}-{number:03d}", clean(message["message"]))
                    )
    elif source == "crosswoz":
        with zipfile.ZipFile(cache / "crosswoz-test.json.zip") as archive:
            dialogues = json.loads(archive.read("test.json"))
        for name in sorted(dialogues, key=int):
            for index, message in enumerate(dialogues[name]["messages"]):
                turns.append((f"{name}-{index:02d}", name, clean(message["content"])))
    else:
        with zipfile.ZipFile(cache / "risawoz-source-data.zip") as archive:
            dialogues = json.load(io.TextIOWrapper(archive.open("source-data/all_test600_new.json"), encoding="utf-8"))
        for dialogue in dialogues:
            name = dialogue["dialogue_id"]
            for turn in dialogue["dialogue"]:
                for side in ("user_utterance", "system_utterance"):
                    turns.append((f"{name}-{turn['turn_id']:02d}{side[0]}", name, clean(turn[side])))
    return turns


# --- typing ------------------------------------------------------------------------------------------


def word_window(gold, rng):
    """`(start, length)` of a word, or of two adjacent words, of `WORD_MIN` to `WORD_MAX` characters
    in `gold`, by jieba's segmentation, chosen at random; None if there is none."""
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
    """What a user types for these syllables: in full, the first letters, or a mix of the two.

    A mixed key has at least one syllable of each kind; which ones is random. An initial is the
    first letter (中国人 `zgr`), the way initials-only input is usually typed.
    """
    if style == "full":
        return "".join(pinyin)
    if style == "initials":
        return "".join(p[0] for p in pinyin)
    while True:
        full = [rng.random() < 0.5 for _ in pinyin]
        if any(full) and not all(full):
            return "".join(p if keep else p[0] for p, keep in zip(pinyin, full, strict=True))


def typed_cases(clause_id, made, rng):
    """The cases one sampled clause gives: the clause and a word inside it, each typed three ways."""
    out = []
    shared = {key: made[key] for key in made if key not in ("gold", "readings", "context")}
    for style in ("full", "initials", "mixed"):
        out.append(
            {
                "id": clause_id if style == "full" else f"{clause_id}.c{style[0]}",
                "level": "clause",
                "style": style,
                "input": typed(made["readings"], style, rng),
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


def build(args):
    cache = Path(args.cache)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    import jieba

    jieba.setLogLevel(60)

    # One generator per source, so that a change to one source's rules leaves the others' samples as they were.
    sampled = gov_clauses(cache, manifest, random.Random(f"{SEED}-gov"))
    pools = {"gov": len(manifest["gov"])}
    for source, spec in DIALOGUES.items():
        rng = random.Random(f"{SEED}-{source}")
        turns = dialogue_turns(cache, source)
        pools[source] = len(turns)
        rng.shuffle(turns)
        sampled += stratified(source, turns, rng, spec["quotas"])

    # Clauses read and dropped by hand (review.tsv): the gold is not the standard written form.
    reviewed = set()
    with open(Path(__file__).resolve().parent / "review.tsv", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                source, unit, gold = line.rstrip("\n").split("\t")[:3]
                reviewed.add((source, unit, gold))

    seen = set()
    rows = []
    duplicates = 0
    counters = {}
    for made in sampled:
        if (made["source"], made["unit"], made["gold"]) in reviewed:
            reviewed.discard((made["source"], made["unit"], made["gold"]))
            continue
        if made["gold"] in seen:
            duplicates += 1
            continue
        seen.add(made["gold"])
        counters[made["source"]] = counters.get(made["source"], 0) + 1
        clause_id = f"{made['source'][0]}-{counters[made['source']]:04d}"
        rows += typed_cases(clause_id, made, random.Random(f"{SEED}-{clause_id}"))

    if reviewed:
        raise SystemExit(f"review.tsv names clauses that were not sampled: {sorted(reviewed)}")
    with open(args.out, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    # The same cases as MSIME's `convert_eval` reads them: id, input, gold, syllables, tags, context.
    with open(args.tsv, "w", encoding="utf-8") as out:
        out.write("# chinese-ime-lm eval/sentences-ctx-v1, written by build_set.py from sampled.jsonl\n")
        for row in rows:
            tags = [row["source"], row["level"], row["style"], "context" if row["context"] else "no-context"]
            fields = [row["id"], row["input"], row["gold"], str(row["syllables"]), ",".join(tags), row["context"]]
            if any("\t" in field or "\n" in field for field in fields):
                raise SystemExit(f"{row['id']}: a tab or a line break in a field")
            out.write("\t".join(fields) + "\n")

    clauses_kept = [row for row in rows if row["level"] == "clause" and row["style"] == "full"]
    by_source = {}
    for row in clauses_kept:
        key = f"{row['source']}/{'context' if row['context'] else 'none'}"
        by_source[key] = by_source.get(key, 0) + 1
    print(
        f"{len(clauses_kept)} clauses ({duplicates} duplicate golds dropped), {len(rows)} typed cases; {by_source}; units available {pools}",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("fetch")
    one.add_argument("--cache", required=True)
    two = sub.add_parser("build")
    two.add_argument("--cache", required=True)
    two.add_argument("--out", required=True, help="sampled.jsonl: every typed case with its source")
    two.add_argument("--tsv", required=True, help="the same cases for MSIME's convert_eval")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(Path(args.cache))
    else:
        build(args)


if __name__ == "__main__":
    main()
