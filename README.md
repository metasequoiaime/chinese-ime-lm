# chinese-ime-lm

A character-level language model that reranks the candidates a Chinese input method produces, with the training pipeline that builds it and the measurements that say when it helps.

It is meant to be used by input methods other than the one it came from. The model file is plain safetensors, the reference implementation depends on nothing but serde, and the part that is specific to any one engine is a predicate the caller supplies.

## What it does, and where it does not help

Reranking is not uniformly good. Measured against the evaluation sets in `eval/`, driven through a real input runtime rather than offline:

| The engine's leading candidate | cases | engine top-1 | reranked top-1 |
|---|---|---|---|
| An exact dictionary hit on the whole key | 2052 | **0.719** | 0.690 at best |
| Assembled by a lattice or fallback decoder | 37 | 0.000 | **0.189** |

A dictionary that answered the whole key already carries corpus frequency. A character model does not have that, and overriding those candidates costs accuracy at every threshold tried — twenty-five weightings of a model score against rank and dictionary evidence, and not one of them beat simply leaving dictionary-led lists alone. Decoder-assembled candidates carry no frequency evidence, and that is where the model earns its place.

So the model is gated, and `Reranker::best_where` takes the gate as a predicate:

```rust
let promoted = reranker.best_where(committed_text, &candidates, |index| {
    // Your engine's answer to: did the dictionary answer the whole key with this one?
    sources[index] == CandidateSource::Dictionary
});
```

## Two things that will silently ruin your measurements

**Only candidates covering the whole key are comparable.** Engines return prefixes alongside full answers. A summed log-probability is larger for fewer characters, so a mixed-length list ranks the shortest candidate first every single time, regardless of quality. Measured naively this reads as the model destroying accuracy; restricted to full-cover candidates, the same weights improve it. Scores here are per character for the same reason.

**The gate the evaluation can apply is not the gate the runtime can.** An evaluation knows the expected answer and can filter by its length; an input method cannot. Measure with the criterion you will actually ship — the leading candidate's source and length — or your numbers will be better than your product.

## Reranking has a ceiling, and it is low

Reranking can only choose among readings the engine already assembled. In the sentence set every case offered exactly two full-cover candidates, so the decision was always between two readings; any error they shared was unreachable.

`examples/decode.rs` removes that ceiling by searching the syllables directly — every character the pinyin table allows, scored by the model. On this model it performs worse than the engine it was meant to improve, because a character model without word-frequency evidence loses on proper nouns and rare collocations. That is a statement about this model's capacity and corpus, not about the approach.

## Training

```sh
pip install -r training/requirements.txt

python training/corpus.py c4   --out data/c4.txt --max-chars 1_000_000_000
python training/corpus.py lccc --out data/lccc.txt --split large

python training/train.py --corpus data/c4.txt data/lccc.txt --out runs/desktop --preset desktop
python training/export.py --run runs/desktop --out dist/model.safetensors --precision int8
```

| Preset | Layers | Width | Context | Vocabulary | Parameters | int8 |
|---|---|---|---|---|---|---|
| `keyboard` | 6 | 256 | 64 | 8192 | 6.8M | 7.1 MB |
| `desktop` | 8 | 448 | 128 | 12288 | 24M | ~24 MB |

`keyboard` is sized to load inside a mobile keyboard extension, which shares a memory budget with the engine and its dictionaries. Capacity is the lever that mattered most: at 5,000 steps the `desktop` preset already reached a lower validation perplexity than `keyboard` did after 50,000.

`corpus.py` also offers `wiki` and `docs` sources. They are not in the commands above on purpose — see the licensing note below.

## Inference

Plain safetensors. Configuration, vocabulary and corpus attribution travel in the `__metadata__` header, so the model is one self-describing file with no sidecars. Embeddings are tied and `head.weight` is absent; a loader reuses `tok.weight`. Under int8, two-dimensional weights are quantized symmetrically per output row with a `<name>.scale` companion in float32.

int8 and float16 disagreed on no ranking decision across 2105 cases, so int8 is the one to ship.

The reference implementation is plain Rust with no unsafe and no numeric dependency. One detail is worth copying if you reimplement it: a dot product accumulating into a single running sum makes every multiply-add wait for the previous one, so the loop runs at the latency of the instruction rather than its throughput. Eight independent partial sums took a nine-candidate decision from 122 ms to 26 ms. Caching the prefix — committed text changes on commit, not on every keystroke — takes a realistic two-candidate decision to 7.5 ms.

## Licensing

The code is Apache-2.0.

Published model weights are trained only on corpora whose licenses carry no share-alike obligation, so that adopting the model does not impose one on you:

- The Chinese portion of C4 — ODC-BY
- LCCC — MIT

Every one of them requires attribution to travel with a model trained on it, and `export.py` writes that into the weights file itself rather than into a document beside it.

`corpus.py` can also build from Chinese Wikipedia (CC BY-SA 4.0) and MDN (CC BY-SA 2.5). Whether model weights are a derivative work of their training text is unsettled, and most open models train on Wikipedia and release permissively — but a resource meant for other projects to adopt should not hand them that question, so published weights leave those sources out. They remain available for anyone who has already decided the question for themselves.

## Evaluation sets

`eval/quanpin-words-v1.tsv` is 25,119 word cases frozen from the MIT-licensed sample dictionary of the MSIME engine. `eval/sentences-v1.tsv` is 60 hand-authored whole-sentence cases. Both are worth more than the model: they are what turns an opinion about a reranking change into a number, and they carry the two traps above as fixtures rather than as advice.
