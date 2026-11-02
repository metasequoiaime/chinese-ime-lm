---
license: apache-2.0
language:
- zh
pipeline_tag: text-ranking
datasets:
- allenai/c4
- HuggingFaceFW/fineweb-2
- silver/lccc
tags:
- chinese
- pinyin
- input-method
- reranking
- character-level
- on-device
- onnx
---

# pinyin-ime-keystroke-reranker-25M

A 24.9M-parameter character-level language model that ranks a Chinese pinyin input method's candidates on every keystroke. It ships as one ONNX Runtime graph in ORT format, 30.8 MB with uint8 weights, plus two small JSON files.

**It does not decode pinyin.** It takes the candidate list your engine assembled and judges which one the preceding text supports. Feeding it a pinyin string will not give you Chinese text.

Trained and measured in [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm). The same bytes are in the GitHub release [`rerank-v1`](https://github.com/metasequoiaime/chinese-ime-lm/releases/tag/rerank-v1), where the files carry a `rerank-8x448.` prefix so that both models fit in one release.

## Which of the two

The two models are tiers by device speed, not alternatives in quality. A host loads this one first, measures a page of heavy load, and switches to the smaller one if that takes more than 10 ms.

| | **this model** (`8x448`) | [pinyin-ime-keystroke-reranker-8M](https://huggingface.co/msime/pinyin-ime-keystroke-reranker-8M) (`6x256`) |
|---|---|---|
| Parameters | 24.9M | 7.96M |
| File (uint8 weights) | 30.8 MB | 8.2 MB |
| ctx-v1, with context (5,468 cases) | **91.8%** | 90.1% |
| ctx-v1, without context (849 cases) | **83.4%** | 82.8% |
| Reading a candidate character | softmax over the whole vocabulary | one dot product (self-normalised in training) |
| Meant for | desktops, recent phones | older phones |

For reference, the 4.25M sentence model ([pinyin-ime-reranker-4M](https://huggingface.co/msime/pinyin-ime-reranker-4M), GitHub release `model-v1`) scores 89.6% with context and 81.3% without on the same set, scored the same way.

**These files do not replace `sentence-model.safetensors`.** The sentence models are a self-describing safetensors file for the pure-Rust [`chinese-ime-lm`](https://crates.io/crates/chinese-ime-lm) crate; this is an ONNX Runtime graph with a different calling convention, described below.

## Files

| File | Bytes | What it is |
|---|---|---|
| `lm_step.ort` | 30,834,688 | The whole keystroke path as one graph, weights quantized dynamically to uint8, optimised ahead of time for a minimal ONNX Runtime build |
| `model.json` | 682 | What a runtime needs besides the graph: cache shape, context window, BOS and UNK ids, whether candidate characters may be read without the softmax, and the corpus attribution |
| `vocab.json` | 85,613 | The token list. Single characters are the model's characters; an id is the position in the list |
| `required_operators_and_types.config` | 984 | The operators and types the graph uses, written by the export (only its comment line, which named a local build path, now reads `lm_step.ort`): the input to ONNX Runtime's `--include_ops_by_config` and `--enable_reduced_operator_type_support` |
| `LICENSE` | — | Apache-2.0 |
| `NOTICE` | — | Corpus attribution, which must travel with the weights |

The three model files keep the names a host looks for: put them in one directory as they are.

## Use it

The graph is `lm_step` in [`neural/choice/model.py`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/neural/choice/model.py), exported by [`neural/choice/export_step.py`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/neural/choice/export_step.py). It holds the weights once and is called in two ways:

- **When text is committed**, run the context through it into an empty cache. The context is character ids opened with BOS (`bos` in `model.json`), at most the last `context_chars` (64) characters; characters not in `vocab.json` map to `unk`. `context_inputs` builds the feed.
- **On each keystroke**, run only the candidate prefixes that are not in the cache yet. Shared prefixes run once, a candidate's last character never runs, and the context's last node predicts every candidate's first character. `step_inputs` builds the feed and returns the extended prefix list for the next keystroke.

| Input | Type | Shape |
|---|---|---|
| `cache_k`, `cache_v` | float32 | [layers, heads, M, head_dim], M may be 0 |
| `cache_h` | float32 | [M, width] |
| `cache_z` | float32 | [M] |
| `nodes`, `positions` | int64 | [N], N may be 0 when every prefix is cached |
| `visible` | bool | [N, M + N], which cached and new nodes each new node sees |
| `candidates` | int64 | [B, Lb], padded |
| `steps` | int64 | [B, Lb], the node predicting each candidate character, indexing the M + N nodes |
| `lengths` | int64 | [B] |

Outputs: `scores` [B], each candidate's mean log-probability per character, and `next_k`, `next_v`, `next_h`, `next_z`, the cache with the new nodes appended. Feed those back on the next keystroke.

Both models together use 27 operators, so ONNX Runtime can be built down to them from the `required_operators_and_types.config` files (statically linked, about 1.5 MB). `export_step.py --golden` types a pool's cases through the ORT file and writes every keystroke's candidates and scores to `golden.jsonl`, for a runtime's own tests to replay.

**The weights are uint8, not int8, on purpose.** ONNX Runtime picks its int8 kernels by CPU feature. With int8 weights, two Macs with and without I8MM picked different candidates on 42 of 6,317 cases; with uint8 weights the two give bit-identical scores.

## Measurements

[`eval/sentences-ctx-v1/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval/sentences-ctx-v1) is 6,317 cases drawn from text outside the training corpora: clauses from government documents published 2026-09-04 to 10-29 and from the test splits of three crowd-sourced dialogue sets, plus a 2 to 4 character word from each clause typed as full pinyin, initials only, and a mix. The gold answer is what the original author wrote, the context is the text before the cursor in the same document, and the candidate pools were recorded from a real input method. A case counts as right when the scorer picks the gold answer out of the same pool. Sources, licences, sampling rules and the overlap check against training corpora are in that directory's README; the comparison against other scorers on the same pools is in [`docs/measurements.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md).

| | with context (5,468) | without context (849) |
|---|---|---|
| **this model** | **91.8%** | **83.4%** |
| pinyin-ime-keystroke-reranker-8M | 90.1% | 82.8% |
| pinyin-ime-reranker-4M (`model-v1`) | 89.6% | 81.3% |

### Latency

Measured by typing key by key in MSIME's real runtime (`rerank_latency`, 643 cases from sentences-v2 and typing-chat-v1, about 8,100 keystrokes, two rounds). Whole-keystroke p95:

| Device | Model used | Keystroke p95 | `model-v1` 4.25M on the same device |
|---|---|---|---|
| Apple M1 Max / M4 Pro (2 threads) | 8x448 | about 4 ms | about 6 ms |
| iPhone 17 (2 threads) | 8x448 | 3.8–4.1 ms | 4.8–5.4 ms |
| iPhone XR (2 threads) | 6x256 | 4.4–4.9 ms | 11.1–11.9 ms |
| Pixel 3 XL (4 threads) | 6x256 | 9.3 ms | 38–40 ms |

**Report percentiles, not means.** Most keystrokes cost little, so a mean spreads the cost of the expensive ones across the cheap ones.

## Shape

From `model.json`:

| | |
|---|---|
| Layers / width / heads | 8 / 448 / 8 (head dimension 56) |
| Positions | 256 |
| Context window | 64 characters |
| Vocabulary | 12,288 characters |
| Activation | exact GELU |
| Candidate characters read | through the full-vocabulary softmax |
| Parameters | 24.9M |
| Precision | uint8 weights, dynamic quantization |

## Corpus, licence and attribution

The **weights are Apache-2.0**, which is not the licence of the training code (that repository is GPL-3.0, and its `reference/` directory is Apache-2.0).

`LICENSE` and `NOTICE` are in this repository rather than only named in this metadata, because Apache-2.0 asks a redistributor to pass both along and a label on a web page is not a copy of either.

Trained on the Chinese portion of [C4](https://huggingface.co/datasets/allenai/c4) (ODC-BY), the Chinese (`cmn_Hani`) portion of [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (ODC-BY) and [LCCC](https://huggingface.co/datasets/silver/lccc) (MIT). **None carries a share-alike obligation**, so adopting this model does not bring one onto you.

The corpora do require attribution to travel with anything trained on them. It is written into the graph's metadata (`attribution`, `license`), which the ORT format keeps, and the same sentence is in `model.json`:

```
Trained on the Chinese portion of C4 (ODC-BY, https://huggingface.co/datasets/allenai/c4), FineWeb-2 (cmn_Hani) (ODC-BY, https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and LCCC (MIT, https://github.com/thu-coai/CDial-GPT).
```

**Preserve both when redistributing.** The attribution is derived from the corpus the run actually used, not declared: the export builds it from the records written beside each fetched corpus and refuses if any source is share-alike.

## Integrity

```
lm_step.ort                           30,834,688 bytes
sha256  ad729f94fc10cae0326269e2f089d7747ee8efae3dd2affe365c2740c59a4a93

model.json                                   682 bytes
sha256  2a92e6aad513335e0800ff24ef7037acb365d0ae9abb7ae5b92ebf241c065a20

vocab.json                                85,613 bytes
sha256  d7fb2e70de9f0c2c65b822a8fbf604655bfef6d0dcd8f1975d73f1cd5ea7f038

required_operators_and_types.config          984 bytes
sha256  e0c2d4cdbe1c6f1ff9d461a8d3066ca9a4ca42bcfe93ecdc107ad3ec991c22ad
```

These files are loaded by a process that sees every keystroke. **Verify the digests before shipping them.** See [`SECURITY.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/SECURITY.md).
