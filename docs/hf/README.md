# Hugging Face 模型卡

这两个文件是发布到 Hugging Face 的仓库 `README.md`，逐字对应：

| 本地文件 | Hugging Face 仓库 | 权重 |
|---|---|---|
| [`pinyin-ime-reranker-4M.md`](pinyin-ime-reranker-4M.md) | `pinyin-ime-reranker-4M` | `sentence-model.safetensors`，4.25M |
| [`pinyin-ime-reranker-25M.md`](pinyin-ime-reranker-25M.md) | `pinyin-ime-reranker-25M` | `sentence-model.safetensors`，24.9M |

放在仓库里而不是只存在于 Hugging Face 上，是因为卡里的每个数字都来自 [`measurements.md`](../measurements.md) 或权重文件自己的 `__metadata__`。改了测量就要改这里，两边同一个 commit 走。

卡里的 `sha256` 和字节数标定的是某一份具体文件。重新导出权重就必须一并更新，否则那两行会变成谎话。
