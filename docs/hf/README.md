# Hugging Face 模型卡

这四个文件是发布到 Hugging Face 的仓库 `README.md`，逐字对应：

| 本地文件 | Hugging Face 仓库 | 仓库内的权重 |
|---|---|---|
| [`pinyin-ime-reranker-4M.md`](pinyin-ime-reranker-4M.md) | [`msime/pinyin-ime-reranker-4M`](https://huggingface.co/msime/pinyin-ime-reranker-4M) | `sentence-model.safetensors`，4.25M，由 `dist/sentence-model.safetensors` 上传 |
| [`pinyin-ime-reranker-25M.md`](pinyin-ime-reranker-25M.md) | [`msime/pinyin-ime-reranker-25M`](https://huggingface.co/msime/pinyin-ime-reranker-25M) | `sentence-model.safetensors`，24.9M，由 `dist/sentence-model-desktop.safetensors` 上传 |
| [`pinyin-ime-keystroke-reranker-25M.md`](pinyin-ime-keystroke-reranker-25M.md) | [`msime/pinyin-ime-keystroke-reranker-25M`](https://huggingface.co/msime/pinyin-ime-keystroke-reranker-25M) | `lm_step.ort`，8x448、24.9M，由 release `rerank-v1` 的 `rerank-8x448.*` 上传 |
| [`pinyin-ime-keystroke-reranker-8M.md`](pinyin-ime-keystroke-reranker-8M.md) | [`msime/pinyin-ime-keystroke-reranker-8M`](https://huggingface.co/msime/pinyin-ime-keystroke-reranker-8M) | `lm_step.ort`，6x256、7.96M，由 release `rerank-v1` 的 `rerank-6x256.*` 上传 |

前两个是整句重排模型（safetensors，`model-v1`），后两个是按键重排模型（ORT 格式，`rerank-v1`），两类文件格式和调用方式都不同，见下面各自一节。

## 整句重排模型（`model-v1`）

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
curl -s https://huggingface.co/api/models/msime/pinyin-ime-reranker-4M/tree/main \
  | python3 -c "import json,sys; print([(e['path'], e.get('size'), (e.get('lfs') or {}).get('oid')) for e in json.load(sys.stdin)])"
```

## Resource lock 条目

`export.py` 末尾打印的那条 lock 条目带一个 `https://example.invalid/` 占位 URL，因为导出时还不知道会发到哪。

**MSIME 那边的真实 lock 用的是 GitHub release 的 URL**，不是下面这两条：`msime/resources/desktop-dictionary.lock.json` 和 `msime/resources/settled-model.lock.json` 指向 `releases/download/model-v1/`。下面两条是同一批字节的 Hugging Face 来源，给不想依赖 GitHub release 的采用方用——**摘要和字节数与 GitHub 那份相同**，因为文件就是同一个。要换源只换 `url`，`sha256` 不动；如果换了 `url` 之后摘要对不上，那说明两边的文件不是同一份，该查的是发布流程而不是改摘要。

```json
{
  "name": "sentence-model.safetensors",
  "url": "https://huggingface.co/msime/pinyin-ime-reranker-4M/resolve/main/sentence-model.safetensors",
  "sha256": "86ac529510cb3b4968a5e6ade83ec8080f5b34a0a681e75c362accbbd387d1c1",
  "size": 4486280
}
```

```json
{
  "name": "sentence-model.safetensors",
  "url": "https://huggingface.co/msime/pinyin-ime-reranker-25M/resolve/main/sentence-model.safetensors",
  "sha256": "0a6ecba69bf1d39c7eb49549c716477dd6c03fdf05757773e1f50435262fb469",
  "size": 25480184
}
```

`resolve/main/...` 会 302 到 CDN，`curl -L` 和 range 请求都正常（实测 200 / 206），所以它可以直接当下载地址用，不需要先解析。

**两条的 `name` 是同一个，所以一份 lock 只能放一条。** 这不是疏漏——`sentence-model.safetensors` 是 host 要找的名字，一个装置只装一个重排模型，选哪个就是选哪条。要让两个模型共存，得先改 host 那边按名字寻址的约定，而不是在这里给文件改名。

这两条的 `sha256` 和 `size` 是从**线上**那份核出来的，不是从本地 `dist/` 抄的：LFS 的 `oid` 与本地摘要逐个相符之后才写在这里。重新导出权重就必须重新核，`url` 不变不代表内容不变。

## 按键重排模型（`rerank-v1`）

GitHub release [`rerank-v1`](https://github.com/metasequoiaime/chinese-ime-lm/releases/tag/rerank-v1) 把两个模型放在一起，所以文件带 `rerank-<尺寸>.` 前缀。分成两个 HF 仓库之后不需要前缀，上传时落成 host 要找的原名，和 `export_step.py` 写出的名字一致：

| release 资产 | HF 仓库内 |
|---|---|
| `rerank-<尺寸>.ort` | `lm_step.ort` |
| `rerank-<尺寸>.model.json` | `model.json` |
| `rerank-<尺寸>.vocab.json` | `vocab.json` |
| `rerank-<尺寸>.required_operators_and_types.config` | `required_operators_and_types.config` |

每个仓库共七个文件：上面四个，加 `README.md`（`docs/hf/pinyin-ime-keystroke-reranker-*.md`）、`LICENSE`（同样是 **`reference/LICENSE`**，理由同上）和 `NOTICE`。`NOTICE` 里已经写了按键模型的署名在 ONNX 图元数据和 `model.json` 里。`required_operators_and_types.config` 是导出时与 `lm_step.ort` 同一次写出的那份，不要用别的导出重新生成——它决定精简构建的 ONNX Runtime 里留下哪些算子和类型，与图不配套会在加载时缺内核。唯一改动的是开头的注释：ONNX Runtime 在那里写下生成它的模型的绝对路径，会带出导出机器上的用户目录，发布前把 `# - <绝对路径>` 那行换成 `# - lm_step.ort`。解析器跳过 `#` 开头的行，算子行一个字节不动；改完用 ONNX Runtime 自己的 `tools/python/util/reduced_build_config_parser.py` 解析新旧两份，算子集合和类型表应当相同。

卡里的数字来自 `rerank-v1` 的 release 说明和 `model.json`，`sha256` 和字节数标定的是 release 里那几份文件。上传前先按 release 说明的「校验」核对下载下来的文件，上传后用上面的 tree 接口核线上的 `size` 和 LFS `oid`（`model.json`、`vocab.json`、`.config` 是小文件，不走 LFS，只核 `size`，再下载下来比摘要）。重新导出模型就要发新的 release，并一并更新这两张卡。

**准确率要在 release 里那几份文件本身上量**，不能拿导出之前的检查点、另一种量化的图或另一次训练的结果顶替。最初的说明和卡就错在这里：两个按键模型的数字量的是导出前另做 int8 量化的两张图（91.8% / 83.4%、90.1% / 82.8%），发布的 uint8 `lm_step.ort` 是 91.9% / 82.7%、90.4% / 83.0%；`model-v1` 那一行写的 89.6% / 81.3% 其实是另一个 6x256 训练中间版本的成绩，`model-v1` 自己是 79.5% / 76.1%。对照模型的数字直接取 `eval/sentences-ctx-v1/results.json` 里那个打分器的条目，口径相同（同一个池、挑按字平均对数概率最高的那个、上文按 `rerank_context` 截、每条用例从空缓存开始）。量 ORT 文件时每条用例都要从空缓存开始：MSIME 的 `OnnxScorer` 在上文不变时沿用上一次的缓存，uint8 动态量化让一起计算的节点影响分数，连着跑整个池时，接在同上文用例后面的那些会有零点几个点的出入。

```sh
hf upload msime/pinyin-ime-keystroke-reranker-25M rerank-8x448.ort lm_step.ort
```
