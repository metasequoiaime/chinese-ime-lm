"""Cases whose context is the typist's own earlier messages, the way a chat input method sees it.

`build_cases.py` takes an LCCC clause's context from the same turn only. In a chat, the input method remembers what the user committed before, one message after another with a line break between them (MSIME's runtime keeps that text as the rerank context), so a clause of the third message reads the first two messages before it. The 2.0 keystroke model never trained on that shape: given 「妈」 and then 「下个月是爸的生日」 it prefers 「下个月十八的生日」, and a message that went wrong takes 「他」 to 「她」 in the next ones.

Here a dialogue's turn `t` (counting from 0) is typed by the speaker who also typed turns `t - 2`, `t - 4`, ...; those turns, joined with line breaks, followed by a line break and the part of turn `t` before the clause, are the context. Only the typist's own turns: the input method does not see the messages it received. Turns 0 and 1 have no earlier turn of their own and are not used.

Everything else is `build_cases.py`: the clause rule, the readings, the typed cases (the clause in full pinyin, and a 2 to 4 character word in full pinyin, initials and mixed), the exclusion of every eval set's golds and 8-character stretches, and the hash split. MSIME's typing-chat-v1 is excluded as well, since it is the chat set these models are judged on. The dialogues are drawn with their own seed, so they are mostly not the ones `build_cases.py` drew.

usage:
  python neural/teacher/build_chat_cases.py --lccc data/raw/lccc_large.jsonl.gz --eval-ctx <sentences-ctx-v1/sampled.jsonl.gz> \\
    --typing-chat <msime>/resources/eval/typing-chat-v1.tsv --out-dir <data>/cases-chat
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_cases
from build_cases import (
    ROOT,
    Exclusion,
    clause_case,
    clauses,
    eval_texts,
    is_traditional_document,
    sha256,
    split_of,
    typed_cases,
)

SEED = "chat-20261103"
SOURCE = "lccc-chat"
QUOTAS = {"dev": 1000, "train": 30000}
PER_DIALOGUE = 2


def write(rows, jsonl_path, tsv_path, header):
    """`build_cases.write`, except that the context's line breaks go into the TSV as a literal `\\n`, which MSIME's `convert_eval` reads back as a line break (the escaping typing-chat's eval sets use); `chat_case` drops turns with a backslash, which that reading has no escape for. The JSON lines keep the real line breaks."""
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(tsv_path, "w", encoding="utf-8") as out:
        out.write(f"# {header}\n")
        for row in rows:
            tags = [row["source"], row["level"], row["style"], "context" if row["context"] else "no-context"]
            context = row["context"].replace("\n", "\\n")
            fields = [row["id"], row["input"], row["gold"], str(row["syllables"]), ",".join(tags), context]
            if any("\t" in field or "\n" in field or "\r" in field for field in fields):
                raise SystemExit(f"{row['id']}: a tab or a line break in a field")
            out.write("\t".join(fields) + "\n")


def typing_chat_texts(path):
    """`(gold, context)` of every message of typing-chat-v1, the context being the messages before it in the same conversation."""
    texts, before = [], {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or line.startswith("id\t") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            doc, text = fields[1], fields[3]
            texts.append((text, before.get(doc, "")))
            before[doc] = before.get(doc, "") + text + "\n"
    return texts


def chat_case(unit, turns, t, rng):
    """A clause of turn `t` with the typist's earlier turns as its context, or None."""
    text = turns[t]
    if not text or is_traditional_document(text):
        return None
    own = [turn for turn in turns[t - 2 :: -2] if turn][::-1]
    if any("\\" in turn or "\t" in turn or "\r" in turn for turn in [*own, text]):
        return None
    if not own:
        return None
    options = list(clauses(text))
    if not options:
        return None
    start, clause = rng.choice(options)
    return clause_case(SOURCE, unit, "\n".join(own) + "\n", text, start, clause)


def sample(split, dialogues, exclusion, taken, rng):
    dropped = {"gold": 0, "gold_stretch": 0, "context_stretch": 0, "duplicate": 0, "untypable": 0}
    out = []
    order = [(index, turns) for index, turns in dialogues if len(turns) >= 3 and split_of(SOURCE, index) == split]
    rng.shuffle(order)
    for index, turns in order:
        if len(out) >= QUOTAS[split]:
            break
        positions = list(range(2, len(turns)))
        rng.shuffle(positions)
        kept = 0
        for t in positions:
            if kept >= PER_DIALOGUE or len(out) >= QUOTAS[split]:
                break
            made = chat_case(f"{index}-{t:02d}", turns, t, rng)
            if made is None:
                dropped["untypable"] += 1
                continue
            reason = exclusion.hit(made)
            if reason:
                dropped[reason] += 1
                continue
            if made["gold"] in taken:
                dropped["duplicate"] += 1
                continue
            taken.add(made["gold"])
            out.append(made)
            kept += 1
    return out, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lccc", required=True, help="lccc_large.jsonl.gz")
    parser.add_argument("--eval-ctx", required=True, help="eval/sentences-ctx-v1/sampled.jsonl.gz")
    parser.add_argument("--typing-chat", required=True, help="MSIME's resources/eval/typing-chat-v1.tsv")
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
    chat = typing_chat_texts(args.typing_chat)
    exclusion = Exclusion()
    for gold, context in texts + chat:
        exclusion.add(gold, context)
    print(f"eval sets: {len(exclusion.golds)} golds, {len(exclusion.stretches)} stretches", file=sys.stderr)

    dialogues, lccc_total = build_cases.lccc_dialogues(args.lccc, random.Random(f"{SEED}-lccc-dialogues"))
    print(f"{len(dialogues)} of {lccc_total} dialogues ({time.time() - started:.0f} s)", file=sys.stderr)

    taken = set()
    report = {"seed": SEED, "quotas": QUOTAS, "per_dialogue": PER_DIALOGUE, "splits": {}}
    for split in ("dev", "train"):
        made, dropped = sample(split, dialogues, exclusion, taken, random.Random(f"{SEED}-{split}"))
        rows = []
        for n, case in enumerate(made, 1):
            clause_id = f"{split[0]}-chat-c-{n:05d}"
            rows += [
                {**row, "split": split} for row in typed_cases(clause_id, case, random.Random(f"{SEED}-{clause_id}"))
            ]
        random.Random(f"{SEED}-{split}-order").shuffle(rows)
        write(
            rows,
            out_dir / f"{split}.jsonl",
            out_dir / f"{split}.tsv",
            f"chinese-ime-lm neural/teacher {split} chat cases, written by build_chat_cases.py",
        )
        kinds = {}
        for row in rows:
            key = f"{row['level']}/{row['style']}"
            kinds[key] = kinds.get(key, 0) + 1
        report["splits"][split] = {
            "clauses": len(made),
            "typed_cases": len(rows),
            "by_kind": dict(sorted(kinds.items())),
            "dropped": dropped,
        }
        print(
            f"{split}: {len(made)} clauses, {len(rows)} cases; dropped {dropped} ({time.time() - started:.0f} s)",
            file=sys.stderr,
        )
        for case in made:
            exclusion.add(case["gold"], case["context"])

    report["inputs"] = {
        "lccc": {"path": args.lccc, "sha256": sha256(args.lccc), "dialogues": lccc_total, "read": len(dialogues)},
        "eval": {path: {"cases": n, "sha256": sha256(path)} for path, n in eval_counts.items()},
        "typing_chat": {"path": args.typing_chat, "messages": len(chat), "sha256": sha256(args.typing_chat)},
    }
    import pypinyin

    report["tools"] = {"pypinyin": pypinyin.__version__, "jieba": jieba.__version__, "python": sys.version.split()[0]}
    (out_dir / "build.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
