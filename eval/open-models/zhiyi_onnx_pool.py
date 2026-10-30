"""Score the candidate pool with Zhiyi's int8 ONNX word-choice model, the way the Zhiyi IME feeds it.

The model is `laya.int8g.onnx` from Zhiyi IME's release `model-zhen-long-l12` (知意 1.4V2-Long-L12,
Apache-2.0), the graph Zhiyi IME 1.5.12-1.5.18 shipped and Qingjian forwards unchanged as
`zhiyi-onnx-v1.0.0`. It is run by ONNX Runtime on the CPU with the session options of Zhiyi's
`laya::Reranker` (4 intra-op threads, 1 inter-op thread, all graph optimizations).

The sequence is Zhiyi's `Reranker::build` for a `ctx-first` model (`unified_vocab.json`
`choice_format`):

    <bos> context[-128 tokens:] <eos> <mask> per unit <eos> <mask> " cand1"[:16] <mask> " cand2"[:16] ... <eos>

The units are `slot_units` of the leading candidate's syllables against the typed letters, as
`pinyin_units` in Zhiyi's `laya_rerank.cc` takes them. The leading candidate's syllables are not in
the pool, so they are recovered the same way as in `qingjian_pool.rs`: split the letters into as many
syllables of the model's unit table as the case has, and when more than one split exists keep the one
the leader's characters can be read as by the model's own `readings`. The pick is the highest logit,
the first one on a tie (Zhiyi sorts stably).

The context is what Zhiyi IME hands the model: the last `laya.context_chars` (128) characters before
the caret (`laya::utf8_tail`), of which the sequence keeps the last `choice_context_tokens` tokens.

A case typed as initials or mixed (`style` in the pool; full pinyin when absent) cannot be split into
syllables of its own. Zhiyi takes the leading candidate's syllables from its engine; here they are
recovered from the leader's characters: readings from the model's `readings` table that
`slot_units` can match against the typed letters, the first in the table's order.

Zhiyi IME itself does not call the model when there is no text before the caret unless its word n-gram
is loaded (`laya.ngram_without_context`), and adds that n-gram, a rank prior and the composer's score to
the model's log-probability. None of those are the model, so none are used here, and the model is
called with an empty context too.

usage:
  python eval/open-models/zhiyi_onnx_pool.py --model-dir <zhiyi-onnx> --pool eval/open-models/pool.jsonl > picks.jsonl
  python eval/open-models/zhiyi_onnx_pool.py --model-dir <zhiyi-onnx> --check <test_vectors.safetensors>
"""

import argparse
import json
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Vocab:
    """The parts of `unified_vocab.json` the word choice reads."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.unit_ids = {name: index for index, name in enumerate(data["units"])}
        self.syllables = sorted(name[2:] for name in data["units"] if name.startswith("f:"))
        self.readings = data["readings"]
        self.max_slots = data.get("max_slots", 24)
        self.context_tokens = data.get("choice_context_tokens", 128)
        self.candidate_tokens = data.get("choice_candidate_tokens", 16)
        if data.get("choice_format") != "ctx-first":
            raise SystemExit("this adapter implements the ctx-first word-choice sequence only")


def splits(keys, count, table):
    """Every way to split `keys` into exactly `count` syllables of `table`."""
    if not keys:
        return [[]] if count == 0 else []
    if count == 0:
        return []
    out = []
    for syllable in table:
        if keys.startswith(syllable):
            out.extend([syllable, *rest] for rest in splits(keys[len(syllable) :], count - 1, table))
    return out


CONTEXT_CHARS = 128  # Zhiyi IME's laya.context_chars


def abbreviated_lead(case, vocab):
    """Readings of the leader's characters that `slot_units` matches against the typed letters."""
    lead = case["candidates"][0]
    code = case["input"]

    def walk(i, pos, chosen):
        if i == len(lead):
            if pos == len(code):
                yield list(chosen)
            return
        for reading in vocab.readings.get(lead[i], []):
            last = i + 1 == len(lead)
            steps = [len(reading)] if code.startswith(reading, pos) else []
            for k in range(len(reading) - 1, 0, -1):
                initial = k == 1 or (k == 2 and reading[1] == "h" and reading[0] in "zcs")
                if (last or initial) and code[pos : pos + k] == reading[:k]:
                    steps.append(k)
            for step in steps:
                chosen.append(reading)
                yield from walk(i + 1, pos + step, chosen)
                chosen.pop()

    for syllables in walk(0, 0, []):
        if slot_units(syllables, code):
            return syllables
    raise SystemExit(f"{case['id']}: no reading of {lead} fits {code}")


def lead_syllables(case, vocab):
    if case.get("style", "full") != "full":
        return abbreviated_lead(case, vocab)
    found = splits(case["input"], case["syllables"], vocab.syllables)
    if len(found) > 1:
        lead = case["candidates"][0]
        found = [s for s in found if all(syl in vocab.readings.get(ch, []) for syl, ch in zip(s, lead, strict=True))]
    if len(found) != 1:
        raise SystemExit(f"{case['id']}: {len(found)} readings of {case['input']} fit the leader")
    return found[0]


def slot_units(syllables, code):
    """Zhiyi's `laya::slot_units`: each syllable as typed, the full syllable tried first."""

    def walk(i, pos, out):
        if i == len(syllables):
            return pos == len(code)
        syllable = syllables[i]
        last = i + 1 == len(syllables)
        if code.startswith(syllable, pos):
            out.append(("f", syllable))
            if walk(i + 1, pos + len(syllable), out):
                return True
            out.pop()
        for k in range(len(syllable) - 1, 0, -1):
            initial = k == 1 or (k == 2 and syllable[1] == "h" and syllable[0] in "zcs")
            if not last and not initial:
                continue
            if code[pos : pos + k] != syllable[:k]:
                continue
            out.append(("p", syllable[:k]))
            if walk(i + 1, pos + k, out):
                return True
            out.pop()
        return False

    out = []
    return out if walk(0, 0, out) else []


class Model:
    def __init__(self, model_dir, threads):
        self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.vocab = Vocab(f"{model_dir}/unified_vocab.json")
        self.cls = self.tokenizer.token_to_id("<bos>")
        self.sep = self.tokenizer.token_to_id("<eos>")
        self.mask = self.tokenizer.token_to_id("<mask>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(f"{model_dir}/laya.int8g.onnx", options, providers=["CPUExecutionProvider"])

    def encode(self, text):
        return self.tokenizer.encode(text.replace("<mask>", " "), add_special_tokens=False).ids

    def sequence(self, context, candidates, units):
        ids = [self.cls]
        context = context[-CONTEXT_CHARS:]
        ids += self.encode(context)[-self.vocab.context_tokens :] if context else []
        ids.append(self.sep)
        unit_ids = [0] * len(ids)
        slots = []
        for kind, text in units[: self.vocab.max_slots]:
            slots.append(len(ids))
            ids.append(self.mask)
            unit_ids.append(self.vocab.unit_ids.get(f"{kind}:{text}", 0))
        ids.append(self.sep)
        unit_ids.append(0)
        markers = []
        for candidate in candidates:
            markers.append(len(ids))
            ids.append(self.mask)
            ids += self.encode(" " + candidate)[: self.vocab.candidate_tokens]
        ids.append(self.sep)
        unit_ids += [0] * (len(ids) - len(unit_ids))
        return ids, unit_ids, markers, slots

    def run(self, ids, unit_ids, markers, slots):
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "unit_ids": np.array([unit_ids], dtype=np.int64),
            "marker_pos": np.array([markers or [0]], dtype=np.int64),
            "marker_mask": np.array([[True] * len(markers) or [False]], dtype=bool),
            "slot_pos": np.array([slots or [0]], dtype=np.int64),
        }
        return self.session.run(["logits"], feed)[0][0][: len(markers)]


# The Chinese word-choice cases of the upstream `test_vectors.safetensors`, as its README describes them
# (the candidates read back from the vectors' token ids): context, typed syllables, candidates.
VECTOR_CASES = {
    0: ("今天天气很好，我们去公园", ["san", "bu"], ["散步", "散布", "三部"]),  # noqa: RUF001 (the text is Chinese)
    1: ("这个问题我需要再", ["si", "kao"], ["思考", "四考", "司考"]),
    2: ("", ["ni", "hao"], ["你好", "拟好", "泥好"]),
    4: ("妈妈问我晚饭吃什么，", ["wo", "jin", "tian"], ["我今天", "我近", "卧"]),  # noqa: RUF001
}


def check(model, path):
    """Check the sequence builder and the forward pass against the upstream `test_vectors.safetensors`.

    The builder has to produce the vectors' token ids, units and marker positions from the plain
    inputs, and the int8 graph's logits have to pick what the fp32 graph's do.
    """
    from safetensors.numpy import load_file

    vectors = load_file(path)
    for case, (context, syllables, candidates) in VECTOR_CASES.items():
        ids, units, markers, slots = model.sequence(context, candidates, [("f", s) for s in syllables])
        same = (
            ids == vectors[f"{case}.input_ids"].tolist()
            and units == vectors[f"{case}.unit_ids"].tolist()
            and markers == vectors[f"{case}.marker_pos"].tolist()
        )
        got = model.run(ids, units, markers, slots)
        want = vectors[f"{case}.logits"]
        print(
            f"case {case}: sequence {'identical' if same else 'DIFFERENT'}, best {int(np.argmax(got))} vs fp32 {int(np.argmax(want))}, max |diff| {float(np.max(np.abs(got - want))):.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="laya.int8g.onnx with tokenizer.json and unified_vocab.json")
    parser.add_argument("--pool", help="eval/open-models/pool.jsonl")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads (Zhiyi's default is 4)")
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--check", help="test_vectors.safetensors from the upstream safetensors package")
    args = parser.parse_args()

    model = Model(args.model_dir, args.threads)
    if args.check:
        check(model, args.check)
        return
    if not args.pool:
        parser.error("--pool is required unless --check is given")

    with open(args.pool, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    inputs = []
    for case in cases:
        units = slot_units(lead_syllables(case, model.vocab), case["input"])
        inputs.append(model.sequence(case.get("context", ""), case["candidates"], units))

    for sequence in inputs:  # untimed: weights paged in, kernels prepared
        model.run(*sequence)
    samples = [[] for _ in cases]
    logits = [None] * len(cases)
    for _ in range(max(1, args.passes)):
        for index, sequence in enumerate(inputs):
            started = time.perf_counter()
            logits[index] = model.run(*sequence)
            samples[index].append((time.perf_counter() - started) * 1000)

    hits = 0
    decisions = []
    for case, scores, times in zip(cases, logits, samples, strict=True):
        pick = int(np.argmax(scores))
        hits += case["candidates"][pick] == case["gold"]
        ms = statistics.median(times)
        decisions.append(ms)
        print(json.dumps({"id": case["id"], "pick": pick, "scores": [float(s) for s in scores], "ms": ms}))
    decisions.sort()
    p95 = decisions[min(len(decisions), max(1, -(-95 * len(decisions) // 100))) - 1]
    print(
        f"zhiyi-onnx: {hits}/{len(cases)} hits; decision p50 {statistics.median(decisions):.2f} ms, p95 {p95:.2f} ms ({args.passes} passes, {args.threads} threads)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
