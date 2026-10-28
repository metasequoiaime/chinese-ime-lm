# chinese-ime-lm

A character-level language model that reranks the candidates a Chinese pinyin input method has already produced. Pure Rust, `unsafe` forbidden, no numeric dependencies: the only dependencies are `serde`, `serde_json` and `thiserror`. Minimum Rust version 1.75.

**It does not decode pinyin.** It takes the candidate list your engine assembled and judges which one the preceding text supports. Feeding it a pinyin string will not give you Chinese text.

This crate is the reference implementation in [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm), where the model is trained and measured.

## Models

The weights are published separately on Hugging Face. Both files are named `sentence-model.safetensors`.

| | [pinyin-ime-reranker-4M](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-4M) | [pinyin-ime-reranker-25M](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M) |
|---|---|---|
| File (int8) | 4.5 MB | 25.5 MB |
| Cost per candidate-character | 0.33 ms | 2.0 ms |
| Worst keystroke, 15-character sentence, 9 candidates | 46 ms | 268 ms |

Take the 4M model for anything that runs on a keystroke, such as a mobile keyboard extension. The model cards carry the accuracy measurements and their limits.

Check a model you downloaded before loading it. From a checkout of the repository, `cargo run --release --example verify -- <model file> --expect-sha256 <digest> --expect-size <bytes>` checks the digest, the size, the licence and the attribution; see [`SECURITY.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/SECURITY.md).

## Use it

```toml
[dependencies]
chinese-ime-lm = "0.2"
```

```rust
use std::sync::Arc;
use chinese_ime_lm::{CandidateFacts, Reranker, SentenceModel};

// Once per process: the weights are read-only and shared by every session.
let bytes = std::fs::read("sentence-model.safetensors")?;
let model = Arc::new(SentenceModel::load(&bytes)?);

// Once per input session: it keeps the committed text's state across keystrokes.
let mut reranker = Reranker::new(Arc::clone(&model));

// On each keystroke, with your engine's candidate list in its own order. The returned index
// refers to that list; `None` means leave the order alone.
let promoted = reranker.best_where(committed_text, &candidates, |index| CandidateFacts {
    // Does this candidate answer the whole key, rather than a prefix of it or a completion
    // running past it? `xian` reads as 现 or as 西安, and both answer the key.
    answers_key: covers_whole_key[index],
    // Should the model defer to this dictionary hit? A hit on a key you corrected carries
    // frequency for the letters that arrived, not for the word the user meant.
    trusted_dictionary_hit: is_dictionary_hit[index] && !corrected_key,
});
```

`SentenceModel` and `Reranker` are both `Send` and `Sync`.

## Gating is part of the model

Reranking is not universally beneficial, and `best_where` enforces two rules rather than leaving them to callers:

- **It never overrides a dictionary hit you trust.** A dictionary that answered the whole key carries corpus word frequency that a character-level model does not have; overriding such hits measurably loses accuracy.
- **It compares only candidates that answer the whole key.** Prefixes and predictive completions answer a different key. Scores are per-character means, so candidates of different lengths that answer the same key are still comparable.

Both facts are things only your engine knows, which is why `CandidateFacts` asks for them. `Reranker::best`, `should_rerank` and `DICTIONARY_SOURCES` infer them from one engine's source numbering and from character count instead; they are kept for that engine, and are wrong for an engine that corrects input. The measurements behind both rules are in [`docs/measurements.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md).

## Other languages

To call this crate from C, C++, Swift, Kotlin or anything else with a C FFI, use its C ABI in [`reference/ffi`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/reference/ffi): a static and a shared library with the header `include/chinese_ime_lm.h`, exporting the same model and the same gating.

To write a native implementation instead, [`docs/format.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/format.md) specifies the file and the forward pass completely; it is the only document needed to write a loader in another language.

## Licence

The code in this crate is licensed under Apache-2.0, unlike the training pipeline in the rest of the repository, which is GPL-3.0. The published weights are also Apache-2.0, and are trained on corpora whose licences require attribution to travel with the model. That attribution is stored in the weights file itself and is returned by `SentenceModel::attribution()`; keep it when you redistribute a model. `SentenceModel::license()` returns the licence the file declares, and is empty for older files that do not declare one.
