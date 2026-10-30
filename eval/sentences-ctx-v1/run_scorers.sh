#!/usr/bin/env bash
# Run every scorer of the comparison on the context set, writing each adapter's picks into $OUT.
#
# Every path below is something you download or build yourself (README.md, "打分器"); nothing here
# fetches anything. A scorer whose path is unset is skipped, so the script can be run a few scorers
# at a time. Each scorer runs on pool.jsonl, on pool-stripped.jsonl (the cases that have a context,
# with it taken away) when it reads the context, and on pool-sentences-v1.jsonl (the 60-case set
# re-recorded with the same engine).
#
#   OUT            where the picks go (summarize.py --picks-dir)
#   PASSES         timing passes per case (default 1, or 5 with TIMING_ONLY=1; a case's time is the median over passes)
#   TIMING_ONLY    set to 1 for the separate timing run (summarize.py --timing-dir): every scorer on pool-timing.jsonl only (timing_pool.py, 480 cases), on a quiet machine
#   PYTHON         a Python with mlx-lm 0.32.2, onnxruntime 1.24.4 and tokenizers
#   OURS_4M        sentence-model.safetensors          OURS_25M  sentence-model-desktop.safetensors
#   QWEN_0_5B QWEN_0_6B QWEN_0_8B QWEN_2B               local snapshot directories of the four mlx-community models
#   QWEN_POOL_STARTS  the Qwen start variants run on pool.jsonl and pool-stripped.jsonl (default `eos`, the best one on the 52-case pool); the others run on pool-sentences-v1.jsonl only. `none endoftext eos` runs all three everywhere, about three times as long.
#   QINGJIAN_POOL  the ime_lm_pool binary built from eval/open-models/qingjian_pool.rs in Qingjian
#   TONGBIAN ZHIWEI                                    the .qjm files
#   ZHIYI_DIR      the zhiyi-v1.0.0 directory (model.safetensors, tokenizer.json, unified_vocab.json)
#   ZHIYI_ONNX PUMPKIN_ONNX                            directories holding laya.int8g.onnx, tokenizer.json, unified_vocab.json
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
: "${OUT:?set OUT to the directory for the picks}"
if [ "${TIMING_ONLY:-}" = 1 ]; then PASSES="${PASSES:-5}"; else PASSES="${PASSES:-1}"; fi
QWEN_POOL_STARTS="${QWEN_POOL_STARTS:-eos}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT"

POOL="$HERE/pool.jsonl"
[ "${TIMING_ONLY:-}" = 1 ] && POOL="$HERE/pool-timing.jsonl"
V1="$HERE/pool-sentences-v1.jsonl"
STRIPPED="$OUT/pool-stripped.jsonl"
FULL_ONLY="$OUT/pool-full-pinyin.jsonl"
"$PYTHON" "$HERE/build_pool.py" --strip "$POOL" --stripped "$STRIPPED"
"$PYTHON" -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as source, open(sys.argv[2], "w", encoding="utf-8") as out:
    for line in source:
        if json.loads(line)["style"] == "full":
            out.write(line)' "$POOL" "$FULL_ONLY"

# name, then the command that takes a pool as its last argument
each() {
  local name=$1 reads_context=$2
  shift 2
  "$@" "$POOL" > "$OUT/$name.jsonl"
  [ "${TIMING_ONLY:-}" = 1 ] && return 0
  if [ "$reads_context" = yes ]; then "$@" "$STRIPPED" > "$OUT/$name.stripped.jsonl"; fi
  "$@" "$V1" > "$OUT/$name.v1.jsonl"
}

if [ -n "${OURS_4M:-}${OURS_25M:-}" ]; then
  cargo build --release --manifest-path "$ROOT/reference/Cargo.toml" --example pool
  POOL_BIN="$ROOT/reference/target/release/examples/pool"
  ours() { "$POOL_BIN" "$1" "$2" --passes "$PASSES"; }
  [ -n "${OURS_4M:-}" ] && each ours-4m yes ours "$OURS_4M"
  [ -n "${OURS_25M:-}" ] && each ours-25m yes ours "$OURS_25M"
fi

qwen() { "$PYTHON" "$ROOT/eval/open-models/mlx_lm_pool.py" --model "$1" --start "$2" --passes "$PASSES" --pool "$3"; }
for spec in "qwen2.5-0.5b:${QWEN_0_5B:-}" "qwen3-0.6b:${QWEN_0_6B:-}" "qwen3.5-0.8b:${QWEN_0_8B:-}" "qwen3.5-2b:${QWEN_2B:-}"; do
  name=${spec%%:*} model=${spec#*:}
  [ -n "$model" ] || continue
  for start in none endoftext eos; do
    case " $QWEN_POOL_STARTS " in
      *" $start "*) each "$name.$start" yes qwen "$model" "$start" ;;
      *) [ "${TIMING_ONLY:-}" = 1 ] || qwen "$model" "$start" "$V1" > "$OUT/$name.$start.v1.jsonl" ;;
    esac
  done
done

if [ -n "${QINGJIAN_POOL:-}" ]; then
  qingjian() { "$QINGJIAN_POOL" "$1" "$2" "$3" --passes "$PASSES"; }
  if [ -n "${TONGBIAN:-}" ]; then
    # 通变 reads complete pinyin and no context: the full-pinyin cases only, nothing to strip.
    qingjian tongbian "$TONGBIAN" "$FULL_ONLY" > "$OUT/hanzhang-tongbian.jsonl"
    [ "${TIMING_ONLY:-}" = 1 ] || qingjian tongbian "$TONGBIAN" "$V1" > "$OUT/hanzhang-tongbian.v1.jsonl"
  fi
  [ -n "${ZHIWEI:-}" ] && each hanzhang-zhiwei yes qingjian zhiwei "$ZHIWEI"
  [ -n "${ZHIYI_DIR:-}" ] && each zhiyi-candle yes qingjian zhiyi "$ZHIYI_DIR"
fi

zhiyi() { "$PYTHON" "$ROOT/eval/open-models/zhiyi_onnx_pool.py" --model-dir "$1" --passes "$PASSES" --pool "$2"; }
[ -n "${ZHIYI_ONNX:-}" ] && each zhiyi-onnx yes zhiyi "$ZHIYI_ONNX"
[ -n "${PUMPKIN_ONNX:-}" ] && each zhiyi-pumpkin yes zhiyi "$PUMPKIN_ONNX"
echo "picks in $OUT"
