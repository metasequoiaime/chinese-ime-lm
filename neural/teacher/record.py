"""Record MSIME's candidates for the sampled cases and cut the pools a scorer chooses from.

MSIME's `convert_eval --dump` types every case into a fresh session after committing its context, with no sentence model in the resource directory, so each list is in the engine's own order. The pools are cut with the context eval set's rules (`eval/sentences-ctx-v1/build_pool.py` on the `eval/large-context-set` branch):

- clauses typed in full pinyin: `eval/open-models/build_pool.py`'s rule, the candidates from the decoder sources `generated` and `fallback` with as many characters as the gold, in the engine's order;
- words, typed any way: the first nine distinct candidates with as many characters as the gold, from any source, in the engine's order (`word_pool`, a copy of the eval set's).

A case whose gold is not in its pool is dropped, and so is a pool of one candidate; `coverage.json` counts both per kind.

A dump keeps every candidate of every reading, about 40 KB a case, so the cases are typed in chunks, several `convert_eval` processes at a time, and each chunk's dump is cut and deleted as soon as it is written. A chunk whose pool file exists is not typed again, so an interrupted run resumes where it stopped.

usage:
  python neural/teacher/record.py --convert-eval <msime>/target/release/examples/convert_eval --resources <resources without the models> \\
    --resource-lock <lock without sentence-model.safetensors> --msime-commit <sha> --cases <data>/cases/dev.jsonl --tsv <data>/cases/dev.tsv \\
    --work <data>/work/dev --out <data>/pools/dev.jsonl --coverage <data>/pools/dev-coverage.json [--chunk 4000] [--jobs 4]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "open-models"))
from build_pool import pool as decoder_pool

WORD_POOL = 9
SOURCE_NAMES = {0: "database", 1: "user", 8: "generated", 9: "fallback"}


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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def cut(dump, cases, pool_path):
    """Cut one chunk's dump into pools; returns the chunk's counts per kind."""
    counts = defaultdict(lambda: {"typed": 0, "anywhere": 0, "kept": 0, "single": 0, "engine_top1": 0})
    lines = []
    with open(dump, encoding="utf-8") as source:
        for line in source:
            case = json.loads(line)
            row = cases[case["id"]]
            if (
                case["gold"] != row["gold"]
                or case["input"] != row["input"]
                or case.get("context", "") != row["context"]
            ):
                raise SystemExit(f"{case['id']}: the dump does not match the cases")
            kind = f"{row['level']}/{row['style']}/{'context' if row['context'] else 'none'}"
            count = counts[kind]
            count["typed"] += 1
            count["anywhere"] += any(candidate["text"] == case["gold"] for candidate in case["candidates"])
            if row["level"] == "clause":
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
            record = {key: row[key] for key in ("id", "split", "level", "style", "source", "unit") if key in row}
            record.update(
                input=case["input"],
                gold=case["gold"],
                syllables=row["syllables"],
                readings=row["readings"],
                context=row["context"],
                candidates=texts,
                sources=sources,
            )
            if "clause" in row:
                record["clause"] = row["clause"]
            lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    # Written whole and renamed, so a pool file that exists is a finished chunk.
    partial = pool_path.with_suffix(".partial")
    partial.write_text("".join(lines), encoding="utf-8")
    partial.rename(pool_path)
    return {key: dict(value) for key, value in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--convert-eval", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--resource-lock", required=True)
    parser.add_argument("--msime-commit", required=True, help="the MSIME commit convert_eval was built from")
    parser.add_argument("--cases", required=True, help="cases jsonl from build_cases.py")
    parser.add_argument("--tsv", required=True, help="the same cases as convert_eval reads them")
    parser.add_argument("--work", required=True, help="chunk files, logs and per-chunk pools")
    parser.add_argument("--out", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--chunk", type=int, default=4000)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    cases = {}
    with open(args.cases, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cases[row["id"]] = row
    with open(args.tsv, encoding="utf-8") as handle:
        header = []
        rows = []
        for line in handle:
            (header if line.startswith("#") else rows).append(line)
    if len(rows) != len(cases):
        raise SystemExit(f"{args.tsv} has {len(rows)} rows, {args.cases} {len(cases)} cases")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, start in enumerate(range(0, len(rows), args.chunk)):
        name = f"chunk-{index:04d}"
        chunks.append(name)
        tsv = work / f"{name}.tsv"
        if not (work / f"{name}.pool.jsonl").exists() and not tsv.exists():
            tsv.write_text("".join(header + rows[start : start + args.chunk]), encoding="utf-8")

    def run(name):
        pool_path = work / f"{name}.pool.jsonl"
        counts_path = work / f"{name}.counts.json"
        if pool_path.exists() and counts_path.exists():
            return name, json.loads(counts_path.read_text(encoding="utf-8")), 0.0
        tsv, dump = work / f"{name}.tsv", work / f"{name}.dump.jsonl"
        started = time.monotonic()
        with open(work / f"{name}.log", "w", encoding="utf-8") as log:
            subprocess.run(
                [
                    args.convert_eval,
                    "--resources",
                    args.resources,
                    "--resource-lock",
                    args.resource_lock,
                    "--set",
                    str(tsv),
                    "--dump",
                    str(dump),
                ],
                stdout=subprocess.DEVNULL,
                stderr=log,
                check=True,
            )
        elapsed = time.monotonic() - started
        counts = cut(dump, cases, pool_path)
        counts_path.write_text(json.dumps(counts), encoding="utf-8")
        dump.unlink()
        tsv.unlink()
        return name, counts, elapsed

    started = time.monotonic()
    totals = defaultdict(lambda: {"typed": 0, "anywhere": 0, "kept": 0, "single": 0, "engine_top1": 0})
    typed_seconds = 0.0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for done, (name, counts, elapsed) in enumerate(pool.map(run, chunks), 1):
            typed_seconds += elapsed
            for kind, count in counts.items():
                for key, value in count.items():
                    totals[kind][key] += value
            print(
                f"{name} done ({done}/{len(chunks)}, {time.monotonic() - started:.0f} s)", file=sys.stderr, flush=True
            )
    wall = time.monotonic() - started

    kept = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for name in chunks:
            text = (work / f"{name}.pool.jsonl").read_text(encoding="utf-8")
            kept += text.count("\n")
            out.write(text)

    resources_line = ""
    log_text = (work / f"{chunks[0]}.log").read_text(encoding="utf-8") if (work / f"{chunks[0]}.log").exists() else ""
    for line in log_text.splitlines():
        if line.startswith("verified evaluation resources"):
            resources_line = line
    lock = json.loads(Path(args.resource_lock).read_text(encoding="utf-8"))
    report = {
        "msime_commit": args.msime_commit,
        "convert_eval_sha256": sha256(args.convert_eval),
        "resource_lock_sha256": sha256(args.resource_lock),
        "dictionary_source_commit": lock.get("source_commit"),
        "dictionary_artifacts": {artifact["name"]: artifact["sha256"] for artifact in lock["artifacts"]},
        "verified": resources_line,
        "cases": len(cases),
        "pools": kept,
        "chunk": args.chunk,
        "jobs": args.jobs,
        # Measured only where a chunk was typed in this run; a resumed run counts the chunks it typed.
        "typing_seconds_summed_over_processes": round(typed_seconds, 1),
        "wall_seconds": round(wall, 1),
        "out_sha256": sha256(args.out),
        "by_kind": {key: totals[key] for key in sorted(totals)},
    }
    Path(args.coverage).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key, count in report["by_kind"].items():
        print(
            f"{key:28s} typed {count['typed']:6d}  gold anywhere {count['anywhere']:6d}  kept {count['kept']:6d}  single {count['single']:5d}  engine top-1 {count['engine_top1']:6d}",
            file=sys.stderr,
        )
    print(f"{kept} pools from {len(cases)} cases in {wall:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
