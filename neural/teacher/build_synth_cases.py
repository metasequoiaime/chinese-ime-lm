"""Cases from the synthetic one-sided chats of `gen_chat.py`: each message typed with the sender's earlier messages of the same conversation as its context.

`build_chat_cases.py` takes the typist's earlier turns from LCCC, open-domain exchanges whose turns rarely build on each other. A generated conversation is one person writing to a parent, a colleague or a landlord, message after message about the same plans, people and places, which is what a chat input method sees and what MSIME's typing-chat-v1 is made of. So every message here is used: its context is the messages before it joined with line breaks, a line break, and the part of the message before the clause (the first message has only that part).

Everything else is `build_chat_cases.py`: the clause rule, the readings, the typed cases (the clause in full pinyin, and a 2 to 4 character word in full pinyin, initials and mixed), the exclusion of every eval set's golds and 8-character stretches (typing-chat-v1 included), and the TSV with `\\n` for a line break. A conversation goes to dev or train whole, by a hash of its id, so no dev context was seen in training. Up to `--per-message` clauses of a message are kept.

The text is synthetic (DeepSeek's `deepseek-v4-pro`, see `gen_chat.py`); a model card for weights trained on it says so.

usage:
  python neural/teacher/build_synth_cases.py --synth <data>/synth/chat.jsonl --eval-ctx <sentences-ctx-v1/sampled.jsonl.gz> \\
    --typing-chat <msime>/resources/eval/typing-chat-v1.tsv --out-dir <data>/cases-synth
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_cases import (
    ROOT,
    Exclusion,
    clause_case,
    clauses,
    eval_texts,
    sha256,
    typed_cases,
)
from build_chat_cases import typing_chat_texts, write

SEED = "synth-20261103"
SOURCE = "synth-chat"
DEV_SHARE = 10


def split_of(conversation_id):
    digest = hashlib.blake2b(f"{SEED}-split-{conversation_id}".encode(), digest_size=8).digest()
    return "dev" if int.from_bytes(digest, "big") % DEV_SHARE == 0 else "train"


def conversation_cases(conversation, per_message, rng):
    """`(clause case, why dropped)` for the clauses of every message, its context the messages before it."""
    messages = conversation["messages"]
    out = []
    for m, text in enumerate(messages):
        if any(ch in text for ch in "\\\t\r\n"):
            out.append((None, "untypable"))
            continue
        before = "".join(earlier + "\n" for earlier in messages[:m])
        options = list(clauses(text))
        rng.shuffle(options)
        for start, clause in sorted(options[:per_message]):
            made = clause_case(SOURCE, f"{conversation['id']}-{m:02d}", before, text, start, clause)
            out.append((made, None if made else "untypable"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth", required=True, help="gen_chat.py's output")
    parser.add_argument("--eval-ctx", required=True, help="eval/sentences-ctx-v1/sampled.jsonl.gz")
    parser.add_argument("--typing-chat", required=True, help="MSIME's resources/eval/typing-chat-v1.tsv")
    parser.add_argument("--sentences-v1", default=str(ROOT / "eval/sentences-v1.tsv"))
    parser.add_argument("--words-v1", default=str(ROOT / "eval/quanpin-words-v1.tsv"))
    parser.add_argument("--open-models-pool", default=str(ROOT / "eval/open-models/pool.jsonl"))
    parser.add_argument("--per-message", type=int, default=2)
    parser.add_argument(
        "--skip-ids", help="a gen_chat.py file whose conversations were already made into cases; they are left out"
    )
    parser.add_argument(
        "--taken",
        action="append",
        default=[],
        help="a cases .jsonl made earlier whose golds are not repeated; repeatable",
    )
    parser.add_argument(
        "--id-tag",
        default="synth",
        help="the middle of the case ids, so a later batch's ids do not collide with an earlier one's",
    )
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

    with open(args.synth, encoding="utf-8") as handle:
        conversations = sorted((json.loads(line) for line in handle if line.strip()), key=lambda c: c["id"])
    if args.skip_ids:
        with open(args.skip_ids, encoding="utf-8") as handle:
            skip = {json.loads(line)["id"] for line in handle if line.strip()}
        conversations = [c for c in conversations if c["id"] not in skip]
    report = {
        "seed": SEED,
        "per_message": args.per_message,
        "dev_share": DEV_SHARE,
        "id_tag": args.id_tag,
        "skip_ids": args.skip_ids,
        "taken": args.taken,
        "splits": {},
    }
    taken = set()
    for path in args.taken:
        with open(path, encoding="utf-8") as handle:
            taken.update(json.loads(line)["gold"] for line in handle if line.strip())
    for split in ("dev", "train"):
        dropped = {"gold": 0, "gold_stretch": 0, "context_stretch": 0, "duplicate": 0, "untypable": 0}
        made = []
        for conversation in conversations:
            if split_of(conversation["id"]) != split:
                continue
            for case, why in conversation_cases(
                conversation, args.per_message, random.Random(f"{SEED}-{conversation['id']}")
            ):
                why = why or exclusion.hit(case) or ("duplicate" if case["gold"] in taken else None)
                if why:
                    dropped[why] += 1
                    continue
                taken.add(case["gold"])
                made.append(case)
        rows = []
        for n, case in enumerate(made, 1):
            clause_id = f"{split[0]}-{args.id_tag}-c-{n:05d}"
            rows += [
                {**row, "split": split} for row in typed_cases(clause_id, case, random.Random(f"{SEED}-{clause_id}"))
            ]
        random.Random(f"{SEED}-{split}-order").shuffle(rows)
        write(
            rows,
            out_dir / f"{split}.jsonl",
            out_dir / f"{split}.tsv",
            f"chinese-ime-lm neural/teacher {split} synthetic chat cases, written by build_synth_cases.py",
        )
        kinds = {}
        for row in rows:
            key = f"{row['level']}/{row['style']}"
            kinds[key] = kinds.get(key, 0) + 1
        conversations_in = sum(1 for c in conversations if split_of(c["id"]) == split)
        report["splits"][split] = {
            "conversations": conversations_in,
            "clauses": len(made),
            "typed_cases": len(rows),
            "by_kind": dict(sorted(kinds.items())),
            "dropped": dropped,
        }
        print(
            f"{split}: {conversations_in} conversations, {len(made)} clauses, {len(rows)} cases; dropped {dropped} ({time.time() - started:.0f} s)",
            file=sys.stderr,
        )

    report["inputs"] = {
        "synth": {
            "path": args.synth,
            "sha256": sha256(args.synth),
            "conversations": len(conversations),
            "models": sorted({c["model"] for c in conversations}),
        },
        "eval": {path: {"cases": n, "sha256": sha256(path)} for path, n in eval_counts.items()},
        "typing_chat": {"path": args.typing_chat, "messages": len(chat), "sha256": sha256(args.typing_chat)},
    }
    import pypinyin

    report["tools"] = {"pypinyin": pypinyin.__version__, "jieba": jieba.__version__, "python": sys.version.split()[0]}
    (out_dir / "build.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
