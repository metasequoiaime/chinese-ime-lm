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

# pinyin-ime-reranker-4M

A 4.25M-parameter character-level transformer that reranks the candidates a Chinese pinyin input method has already produced. 4.5 MB on disk, int8, one self-describing safetensors file with no sidecars.

**It does not decode pinyin.** It takes the candidate list your engine assembled and judges which one the preceding text supports — the gap between "the engine's first guess" and "what the person meant". Feeding it a pinyin string will not give you Chinese text.

Trained and measured in [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm).

## Which of the two

| | **this model** | [pinyin-ime-reranker-25M](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M) |
|---|---|---|
| Parameters | 4,250,112 | 24,863,104 |
| File (int8) | 4.5 MB | 25.5 MB |
| Sentence eval, top-1 | 49 / 56 | **52 / 56** |
| Per-keystroke p95 | **8.6 ms** | 97.2 ms (older measurement, see that card) |

Take this one for anything latency-sensitive — a mobile keyboard extension, or any path that runs on a keystroke. The larger model is more accurate and reaches the ceiling of what reranking can do, but at the cost of a per-keystroke budget that does not fit inside a frame.

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
use chinese_ime_lm::{Reranker, SentenceModel};

let bytes = std::fs::read("sentence-model.safetensors")?;
let model = Arc::new(SentenceModel::load(&bytes)?);
let mut reranker = Reranker::new(model);

// Your engine's candidate list, in its own order. The returned index refers to that list;
// `None` means leave the order alone.
let promoted = reranker.best_where(committed_text, &candidates, |index| {
    // Answered by your engine: did a dictionary hit answer the *whole* key at this position?
    sources[index] == CandidateSource::Dictionary
});
```

For any other language, [`docs/format.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/format.md) is a complete specification of the file and the forward pass — it is the only document needed to write a loader. Two details bite everyone who reimplements it: GELU must be the exact erf form, not the tanh approximation, and the first character of a candidate is predicted from the **last position of the prefix**, so scoring that starts at the candidate's own first position silently leaves that character unscored.

## Gating is part of the model

Reranking is not universally beneficial, and the crate enforces two rules rather than leaving them to callers.

**Never override an exact dictionary hit on the whole key.** A dictionary that answered the entire key already carries corpus word frequency, which a character-level model does not have. Measured on 2,052 such cases: the engine's own first choice is 0.719, and the best reranked result at any threshold tried is 0.690. Twenty-five weightings of model score against rank against dictionary evidence were swept; none beat "leave dictionary hits alone".

**Only candidates covering the whole key are comparable.** Engines also return prefixes and predictive completions. A summed log-probability is larger for fewer characters, so a mixed-length list ranks the shortest candidate first every time, regardless of quality — measured this way, the same weights appear to wreck accuracy. Scores here are per-character means for that reason, and `best_where` additionally restricts comparison to candidates matching the leader's length.

A related trap costs you your measurements rather than your accuracy: a gate that filters by the gold answer's length is available offline and not at runtime. Measure with the predicate you will actually ship, or your numbers will be better than your product.

## Measurements

60 hand-written sentence cases, driven through a real input method runtime rather than scored offline. 56 fall in the decoder bucket, where reranking applies at all.

| | cases / 56 | |
|---|---|---|
| Engine's own first choice | 41 | 0.732 |
| **+ this model** | **49** | **0.875** |
| + 25M model | 52 | 0.929 |
| Gold answer present in the candidate list | 52 | 0.929 |

That last row is the ceiling for any reranker: in the remaining 4 cases the correct reading was never assembled, so no reranker can reach it.

**Rescues and breakages are reported separately**, because they are not the same thing to a user. This model rescues 10 cases the engine got wrong and breaks 2 it got right; the 25M model rescues 11 and breaks 0. So of the 3-case gap between them, only 1 is a rescue this model missed — the other 2 are cases it actively broke. A single total reads as "three fewer improvements" and hides the distinction.

**On the same 52 reachable cases, no pair of models tested reaches significance.** McNemar exact test on this model against the 25M is 0 wins to 3, p = 0.250; against Qwen3-0.6B it is 3 to 3, p = 1.000. Pairwise disagreement is only 2 to 6 cases. Detecting the 1 to 2pp differences that model iteration actually cares about would need roughly one thousand to ten thousand cases at this disagreement rate. Twelve of the 60 sentences also begin with `ta` and take 他 as gold, which is not decidable from pinyin at all — those should never have been collected. **Do not read the accuracy column as a ranking of quality.** Size and latency are deterministic measurements; the accuracy differences are not.

Full analysis, including a comparison against four open Qwen models on the same candidate pool, is in [`docs/measurements.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md).

### Latency and memory

Measured on Apple Silicon through the real keystroke path: 60 sentences, 1,320 keystrokes, paired.

| | p95 | keystrokes over a 16 ms frame | slowest |
|---|---|---|---|
| Rescoring each candidate from scratch | 20.64 ms | 129 / 1320 (9.8%) | 40.76 ms |
| **Resuming by position across keystrokes** | **8.61 ms** | **1 / 1320 (0.1%)** | 16.22 ms |

61% of the characters scored on a keystroke are a prefix of something the previous keystroke already scored, so the reference implementation keeps per-position state and recomputes only what is new. Both evaluation sets produce byte-identical output with and without it, though it is not an identity transform — regrouping a float sum changes the last bits.

Resident memory is not the file size. The loader keeps quantized matrices as `i8` and lifts the per-row scale out of the dot product, which is the same arithmetic rather than an approximation: 24.5 MB resident when expanded to f32, **12.3 MB** kept int8, for about 8% more latency. That trade matters where memory is the hard constraint — an iOS keyboard extension gets a few tens of MB for the engine, the dictionaries and the UI as well.

**Report percentiles, not means.** In the input method this model came from, the mean is 7.9 ms while 25.2% of keystrokes during sentence input exceed one frame; most keystrokes never trigger reranking at all, and a mean spreads the cost of the ones that do across the ones that don't.

## Shape and training

| | |
|---|---|
| Layers / width / heads | 6 / 192 / 4 |
| Context | 64 characters |
| Vocabulary | 8,192 characters (99.987% coverage), tied embeddings |
| Parameters | 4,250,112 |
| Precision | int8, symmetric per output row |
| Validation loss | 4.274 |
| Training steps | 11,500 |

Six shapes were swept under the same corpus and step budget. Five of them in the 4–7M range differ by at most one case, so accuracy cannot separate them and cost decides; this is the cheapest of the five. Only the 24.9M shape crosses that plateau. There is no intermediate result for an intermediate size — a model is either on the plateau or at the ceiling.

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
sentence-model.safetensors  4,486,280 bytes
sha256  86ac529510cb3b4968a5e6ade83ec8080f5b34a0a681e75c362accbbd387d1c1
```

A model file is an untrusted binary that an input method loads into its own process. Verify the digest before shipping one. The reference loader treats every dimension in the header as hostile input and bounds it before multiplying anything out; see [`SECURITY.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/SECURITY.md).

## Evaluation sets

The evaluation sets are arguably more useful than the weights: 25,119 word-level cases and 60 sentence cases, in [`eval/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval). Both traps above are frozen into them as fixed cases rather than left as advice in a document.
