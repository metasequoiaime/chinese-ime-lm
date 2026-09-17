# Training

See the repository README for what the model is for, where it does not help, and the licensing of the corpora.

`corpus.py` builds a normalized corpus of one Chinese sentence per line. `train.py` derives a vocabulary, packs tokens into a memory-mapped array and trains. `export.py` writes safetensors and prints the resource-lock entry an installer needs. `rerank_eval.py` scores a recorded candidate list and reports the two buckets separately.
