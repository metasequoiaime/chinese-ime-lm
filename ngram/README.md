# n-gram 路线

`viterbi.py` 在拼音音节序列上做 Viterbi 解码，用 KenLM 的 n-gram 概率打分。它读取 `model/` 下的 KenLM 二进制模型和 `data/output/` 下的单字拼音表，两者都不入库。

模型的构建流程见仓库根目录 README 的「N-gram info counting」一节。

神经模型一侧的等价物是 `reference/examples/decode.rs`，两者用同一套 `eval/` 评测集衡量，可以直接比较。

## 依赖

根目录的 `requirements.txt`（`kenlm`）。
