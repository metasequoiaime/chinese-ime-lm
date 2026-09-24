---
license: apache-2.0
language:
- zh
pipeline_tag: text-ranking
datasets:
- allenai/c4
- silver/lccc
tags:
- chinese
- pinyin
- input-method
- reranking
- character-level
- on-device
---

# pinyin-ime-reranker-25M

A 24.9M-parameter character-level transformer that reranks the candidates a Chinese pinyin input method has already produced. 25.5 MB on disk, int8, one self-describing safetensors file with no sidecars.

**It does not decode pinyin.** It takes the candidate list your engine assembled and judges which one the preceding text supports — the gap between "the engine's first guess" and "what the person meant". Feeding it a pinyin string will not give you Chinese text.

Trained and measured in [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm).

## Read this before choosing it

This is the accurate one, and on the sentence evaluation set it reaches the ceiling of what any reranker can do. It is also **six times more expensive per decision** than its 4.25M sibling, measured directly on both.

| | **this model** | [pinyin-ime-reranker-4M](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-4M) |
|---|---|---|
| Parameters | 24,863,104 | 4,250,112 |
| File (int8) | 25.5 MB | **4.5 MB** |
| Sentence eval, top-1 | **52 / 56** | 49 / 56 |
| Cost per candidate-character | 2.0 ms | **0.33 ms** |
| Worst keystroke, 15-character sentence, 9 candidates | 268 ms | **46 ms** |

**If the model runs on a keystroke, take the smaller one.** A mobile keyboard extension cannot afford this file or this latency. Use this one on the desktop, on paths that are not per-keystroke, or as the accuracy reference you measure the small one against.

Both files are named `sentence-model.safetensors`, which is the name a host looks for.

## Use it

The reference implementation is Rust, no `unsafe`, no dependencies beyond serde. It is not on crates.io — the [`reference/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/reference) directory is separately licensed Apache-2.0 so that it can be vendored on its own, which is how it is meant to be adopted.

```toml
[dependencies]
# the repository's `reference/` directory, copied into your tree
chinese-ime-lm = { path = "vendor/chinese-ime-lm" }
```

```rust
use std::sync::Arc;
use chinese_ime_lm::{CandidateFacts, Reranker, SentenceModel};

let bytes = std::fs::read("sentence-model.safetensors")?;
let model = Arc::new(SentenceModel::load(&bytes)?);
let mut reranker = Reranker::new(model);

// Your engine's candidate list, in its own order. The returned index refers to that list;
// `None` means leave the order alone. Both facts are things only your engine can answer.
let promoted = reranker.best_where(committed_text, &candidates, |index| CandidateFacts {
    // Does this candidate answer the whole key, rather than a prefix of it or a completion
    // running past it? Not "is it the same length as the leader": `xian` reads as 现 or as
    // 西安 and both answer the key.
    answers_key: covers_whole_key[index],
    // Should the model defer to this dictionary hit? If you corrected the input, a hit on the
    // uncorrected reading carries frequency for the letters that arrived, not for the word
    // the user meant — so the answer there is no.
    trusted_dictionary_hit: sources[index] == CandidateSource::Dictionary && !corrected_key,
});
```

For any other language, [`docs/format.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/format.md) is a complete specification of the file and the forward pass — it is the only document needed to write a loader. Two details bite everyone who reimplements it: GELU must be the exact erf form, not the tanh approximation, and the first character of a candidate is predicted from the **last position of the prefix**, so scoring that starts at the candidate's own first position silently leaves that character unscored.

## Gating is part of the model

Reranking is not universally beneficial, and the crate enforces two rules rather than leaving them to callers.

**Never override a dictionary hit you trust.** A dictionary that answered the entire key already carries corpus word frequency, which a character-level model does not have. Measured on 2,052 such cases: the engine's own first choice is 0.719, and the best reranked result at any threshold tried is 0.690. Twenty-five weightings of model score against rank against dictionary evidence were swept; none beat "leave dictionary hits alone".

That rule has a premise — the key the dictionary answered is the key the user meant. **An engine that corrects input breaks the premise and has to say so.** Type `gongsi` meaning `gongshi` and 公司 is an exact hit on the letters that arrived, carrying frequency for a word nobody asked for. So `trusted_dictionary_hit` is not "did a dictionary match" but "should the model defer to this", and only the caller knows.

**Only candidates answering the whole key are comparable.** Engines also return prefixes and predictive completions, and those are not alternatives to the leader — they answer a different key. A summed log-probability is also larger for fewer characters, so scores here are per-character means.

**Answering the key is not the same as being the same length, and the caller has to say which.** Length agrees with the rule while one key has one segmentation and diverges exactly where correction matters: `xian` reads as 现 or as 西安, both consuming the key, and a length filter keeps only whichever matches the leader. `best_where` therefore asks for `CandidateFacts { answers_key, trusted_dictionary_hit }` rather than inferring either. `Reranker::best` still infers both from the source numbering and the character count, which is fine for an engine that does not correct input and wrong for one that does.

A related trap costs you your measurements rather than your accuracy: a gate that filters by the gold answer's length is available offline and not at runtime. Measure with the predicate you will actually ship, or your numbers will be better than your product.

## Measurements

60 hand-written sentence cases, driven through a real input method runtime rather than scored offline. 56 fall in the decoder bucket, where reranking applies at all.

| | cases / 56 | |
|---|---|---|
| Engine's own first choice | 41 | 0.732 |
| + 4.25M model | 49 | 0.875 |
| **+ this model** | **52** | **0.929** |
| Gold answer present in the candidate list | 52 | 0.929 |

The last two rows being equal is the whole result. 52 is the ceiling for any reranker — in the other 4 cases the correct reading was never assembled, so nothing that reorders a list can reach it (gold 李明在北京工作, every candidate offers 利民). This model takes every reachable case, which means **there is no headroom left on this path**: the next improvement has to come from candidate generation, not from a better reranker.

It rescues 11 cases the engine got wrong and breaks 0 that the engine got right. The 4.25M model rescues 10 and breaks 2, so of the 3-case gap only 1 is a rescue it missed and 2 are cases it broke. Keeping those apart matters because a single total reads as "rescued fewer", while what a user experiences is "something that was already right became wrong".

### The 52 / 52 is not a quality claim

Two things limit it, and both point the same way.

**No pair of models tested reaches significance on these cases.** McNemar exact test, 52 reachable cases: this model against the 4.25M is 3 wins to 0, p = 0.250 — three same-direction calls, the same probability as three coin flips landing heads. Against Qwen3-0.6B, Qwen3.5-0.8B and Qwen3.5-2B it is 3–0, 3–0 and 2–0, p = 0.250, 0.250 and 0.500. Pairwise disagreement is 2 to 6 cases out of 52. At that disagreement rate, detecting the 1 to 2pp differences model iteration cares about needs roughly one thousand to ten thousand cases.

**Part of the score is corpus prior, not ranking ability.** Twelve of the 60 sentences begin with `ta` and take 他 as gold. 他, 她 and 它 are indistinguishable in pinyin, so with no prior context there is information-theoretically no correct answer — a model can only follow its training prior. This model gets them because 他 outranks 她 in C4 and LCCC, which happens to agree with whoever wrote the cases. By the project's own collection rule those 12 should never have been included, and removing them leaves 40 genuinely decidable cases.

So: treat 52 / 52 as "this model does not break anything on a small, partly undecidable set", not as an accuracy figure to defend. **Size and latency are deterministic measurements with no confidence interval; the accuracy column is not.** Full analysis, including a comparison against four open Qwen models on the same candidate pool, is in [`docs/measurements.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md).

### Latency and memory

Cost is close to linear in both candidate count and characters, so one number covers it: **2.0 ms per candidate-character, against 0.33 ms for the 4.25M** — a factor of six. Both were swept twice on one Apple M4 Pro with `reference/examples/bench.rs`, and only cells that agreed across runs are quoted; all 32 of this model's agreed within 3.0%.

One decision, p50 in milliseconds:

| candidates \ characters | 1 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|
| 1 | 3.52 | 9.43 | 17.33 | 25.16 | 31.10 |
| 9 | 19.57 | 73.71 | 145.41 | 216.18 | 270.15 |

**A sentence does not cost one decision, it costs one per prefix length**: typing n characters pays for lengths 1 through n. With 9 candidates, typing 15 characters totals 2,160 ms here against 366 ms for the 4.25M, and the worst single keystroke is 268 ms against 46 ms.

**That is what rules this model off the keystroke path**, and it needs no percentile to see: a single 9-candidate decision at six characters is 109 ms, and a frame is 16 ms.

**The older real-runtime figure is a different harness and is not comparable to the table above.** Driven through an actual input method, rescoring every candidate from scratch on every keystroke, this model's p95 was 97.2 ms with 40.4% of keystrokes past a frame, against 20.3 ms and 10.1% for the 4.25M. The reference implementation later gained cross-keystroke resumption by position, which cut the 4.25M's p95 from 20.64 ms to 8.61 ms over the same 1,320 keystrokes. This model has never been re-measured that way, because doing so needs the recorded candidate lists from that runtime rather than a synthetic sweep. Expect an improvement of a similar shape and do not assume a number.

`bench.rs` calls `SentenceModel::score` and never builds a `Reranker`, so the table above is the no-resumption cost by construction — the same regime as the 97.2 ms and 20.3 ms figures, not the 8.61 ms one.

Resident memory is not the file size. The loader keeps quantized matrices as `i8` and lifts the per-row scale out of the dot product, which is the same arithmetic rather than an approximation. The older expand-to-f32 loader turned this 25.5 MB file into **126 MB resident**; keeping int8 roughly halved that on the 4.25M model it was measured against (24.5 MB to 12.3 MB, for about 8% more latency), but the equivalent number for this model was not recorded.

**Report percentiles, not means.** Most keystrokes never trigger reranking, so a mean spreads the cost of the ones that do across the ones that don't.

## Shape and training

| | |
|---|---|
| Layers / width / heads | 8 / 448 / 8 |
| Context | 128 characters |
| Vocabulary | 12,288 characters (99.999% coverage), tied embeddings |
| Parameters | 24,863,104 |
| Precision | int8, symmetric per output row |
| Validation loss | 3.851 |
| Training steps | 11,000 |

This is the `desktop` preset in `neural/model.py`. Six shapes were swept under the same corpus and step budget; this is the only one that crossed the 4–7M plateau, and it paid for it in latency — the measured factor against the published 4.25M is in the section above. There is no intermediate result for an intermediate size: a model is either on the plateau or at the ceiling.

**Capacity is the most effective lever available.** Under the same corpus, this shape's perplexity at 5,000 steps is already below what the 6×256 keyboard shape reaches at 50,000.

int8 and float16 pick the same candidate on all 2,105 cases where a decision is made, so int8 is what gets released.

## Corpus, licence and attribution

The **weights are Apache-2.0**, which is not the licence of the training code (that repository is GPL-3.0, and its `reference/` directory is Apache-2.0). A great many open-source input methods are MIT, Apache or BSD licensed and cannot link GPL code; a public resource nobody can adopt is not one.

`LICENSE` and `NOTICE` are in this repository rather than only named in this metadata, because Apache-2.0 asks a redistributor to pass both along and a label on a web page is not a copy of either.

Trained on the Chinese portion of [C4](https://huggingface.co/datasets/allenai/c4) (ODC-BY) and [LCCC](https://huggingface.co/datasets/silver/lccc) (MIT). **Neither carries a share-alike obligation**, which is the point: adopting this model does not bring one onto you. Chinese Wikipedia and MDN are both supported by the corpus pipeline and were deliberately not used, because CC BY-SA would.

Both corpora do require attribution to travel with anything trained on them. That attribution is written into the weights file's own `__metadata__` header, so it cannot be separated from the weights:

```
Trained on the Chinese portion of C4 (ODC-BY, https://huggingface.co/datasets/allenai/c4)
and LCCC (MIT, https://github.com/thu-coai/CDial-GPT).
```

**Preserve that field when redistributing.** It is derived from the corpus the run actually used, not declared: each fetched corpus gets a record written beside it, training records the corpus paths, and export refuses rather than guessing if either is missing. The `license` and `attribution` keys are readable at runtime through `SentenceModel::license()` and `SentenceModel::attribution()`.

Corpus lines were deduplicated. On C4 that removed half the characters and cut lines hitting gambling and SEO keyword-farm vocabulary from about 22% to 3.4% — duplication and junk turned out to be the same problem.

## Integrity

```
sentence-model.safetensors  25,480,184 bytes
sha256  0a6ecba69bf1d39c7eb49549c716477dd6c03fdf05757773e1f50435262fb469
```

A model file is an untrusted binary that an input method loads into its own process. Verify the digest before shipping one. The reference loader treats every dimension in the header as hostile input and bounds it before multiplying anything out; see [`SECURITY.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/SECURITY.md).

## Evaluation sets

The evaluation sets are arguably more useful than the weights: 25,119 word-level cases and 60 sentence cases, in [`eval/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval). Both traps above are frozen into them as fixed cases rather than left as advice in a document.
