# Hugging Face 模型卡

这两个文件是发布到 Hugging Face 的仓库 `README.md`，逐字对应：

| 本地文件 | Hugging Face 仓库 | 仓库内的权重 |
|---|---|---|
| [`pinyin-ime-reranker-4M.md`](pinyin-ime-reranker-4M.md) | [`metasequoiaime/pinyin-ime-reranker-4M`](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-4M) | `sentence-model.safetensors`，4.25M，由 `dist/sentence-model.safetensors` 上传 |
| [`pinyin-ime-reranker-25M.md`](pinyin-ime-reranker-25M.md) | [`metasequoiaime/pinyin-ime-reranker-25M`](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M) | `sentence-model.safetensors`，24.9M，由 `dist/sentence-model-desktop.safetensors` 上传 |

两个仓库里的权重都叫 `sentence-model.safetensors`，那是 host 要找的名字。桌面版在本地叫 `-desktop` 只是为了让两份文件能共处一个 GitHub release；分成两个 HF 仓库之后不需要这层改名，所以上传时用 `hf upload <repo> <本地路径> sentence-model.safetensors` 的第三个参数落成正确的名字。

每个仓库另有 `LICENSE` 和 `NOTICE`，四个文件构成一个完整的仓库：

| 仓库内 | 来自 | 为什么必须在 |
|---|---|---|
| `README.md` | `docs/hf/pinyin-ime-reranker-*.md` | card |
| `sentence-model.safetensors` | `dist/` | 权重，署名焊在 `__metadata__` 里 |
| `LICENSE` | **[`reference/LICENSE`](../../reference/LICENSE)（Apache-2.0）** | Apache-2.0 第 4(a) 条要求把许可正文交给接收方；frontmatter 的 `license:` 只是个标签 |
| `NOTICE` | [`NOTICE`](../../NOTICE) | 第 4(d) 条要求 NOTICE 随再分发传播 |

**`LICENSE` 传的是 `reference/LICENSE`，不是根目录那个。** 根目录的 `LICENSE` 是 GPL-3.0，只管仓库里的脚本；把它传上去会给权重挂上一个本项目整套语料政策都在规避的传染性义务，正是 `neural/export.py` 里那段注释在讲的那个 bug。

放在仓库里而不是只存在于 Hugging Face 上，是因为卡里的每个数字都来自 [`measurements.md`](../measurements.md) 或权重文件自己的 `__metadata__`。改了测量就要改这里，两边同一个 commit 走。

卡里的 `sha256` 和字节数标定的是某一份具体文件。重新导出权重就必须一并更新，否则那两行会变成谎话。校验线上那份是否就是这一份：

```sh
curl -s https://huggingface.co/api/models/metasequoiaime/pinyin-ime-reranker-4M/tree/main \
  | python3 -c "import json,sys; print([(e['path'], e.get('size'), (e.get('lfs') or {}).get('oid')) for e in json.load(sys.stdin)])"
```

`export.py` 末尾打印的 resource lock 条目要的 `--url` 就是这两个仓库的 `resolve/main/sentence-model.safetensors`。
