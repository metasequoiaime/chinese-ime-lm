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

一个 24.9M 参数的字级语言模型，在每次按键时给中文拼音输入法的候选排序。发布形式是一张 ORT 格式的 ONNX Runtime 图，uint8 权重，30.8 MB，外加两个小 JSON 文件。

**它不做拼音解码。** 它拿到的是你的引擎已经拼好的候选列表，判断上文支持其中哪一个。喂给它一串拼音，得不到中文。

训练和测量都在 [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm)。这里是第二版，同一批字节也在 GitHub release [`rerank-v2`](https://github.com/metasequoiaime/chinese-ime-lm/releases/tag/rerank-v2) 里，那里的文件带 `rerank-8x448.` 前缀，好让两个模型放进同一个 release。第一版仍在 [`rerank-v1`](https://github.com/metasequoiaime/chinese-ime-lm/releases/tag/rerank-v1)。

## 这一版（`rerank-v2`）

第二版专门针对聊天。输入法在聊天里看到的上文，是同一个人前面发出的几条消息，中间隔着换行；v1 训练时没见过这种上文，所以上文是「妈」时，它仍把「你和爸」打成「喝吧」。

v2 从 v1 接着训练，训练题是 1,352 段合成的单人连发聊天（DeepSeek `deepseek-v4-pro` 为本项目生成），每条消息以同一段里在它之前的消息为上文，共 39,410 个候选池，由 9B 老师打分。只在聊天上继续训会让通用评测掉分，所以训练完再和 v1 做权重插值，本模型取新权重的七成。结构、文件格式和用法都与 v1 相同，`vocab.json` 和 `required_operators_and_types.config` 逐字节不变，可以原样替换。完整过程见仓库里 [`docs/choice-model.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/choice-model.md) 的「聊天上文」一节。

| | v1 | **v2（本仓库）** |
|---|---|---|
| MSIME 带上文的单人聊天（330 段） | 80.6% | **83.3%** |
| MSIME 无上文的聊天（330 段） | 80.0% | **82.7%** |
| MSIME sentences 四套（1,580 句） | 87.8% | 87.8% |
| ctx-v1 有上文（5,468 条） | **91.9%** | 91.4% |
| ctx-v1 无上文（849 条） | 82.7% | 82.7% |

MSIME 的三行在同一套引擎和资源里测，开拼音纠错，整句池 32 条，按第一次按空格上屏是否正确计。和 v1 逐条配对比较，带上文的聊天多对 11 段、少对 2 段（p = 0.02）。ctx-v1 两行见下文「测量」：有上文低了 0.5 个点，是这次为聊天付的代价，仍在 90% 以上。

## 两个里选哪个

两个模型是按设备快慢分的档，不是质量上的二选一。host 先加载这个，测一页重负载，超过 10 ms 就换成小的那个。

| | **本模型**（`8x448`） | [pinyin-ime-keystroke-reranker-8M](https://huggingface.co/msime/pinyin-ime-keystroke-reranker-8M)（`6x256`） |
|---|---|---|
| 参数量 | 24.9M | 7.96M |
| 文件（uint8 权重） | 30.8 MB | 8.2 MB |
| ctx-v1，有上文（5,468 条） | **91.4%** | 90.1% |
| ctx-v1，无上文（849 条） | 82.7% | **83.0%** |
| 读候选字的方式 | 对整个字表做 softmax | 一次点积（训练时自归一化） |
| 适用于 | 桌面、新手机 | 老手机 |

作为参照，4.25M 的整句模型（[pinyin-ime-reranker-4M](https://huggingface.co/msime/pinyin-ime-reranker-4M)，GitHub release `model-v1`）在同一套评测集、同一个候选池上，按同样方式打分，有上文 79.5%，无上文 76.1%。

**这些文件不能替代 `sentence-model.safetensors`。** 整句模型是给纯 Rust 的 [`chinese-ime-lm`](https://crates.io/crates/chinese-ime-lm) crate 用的自描述 safetensors 文件；这里是一张 ONNX Runtime 图，调用方式不同，见下文。

## 文件

| 文件 | 字节数 | 是什么 |
|---|---|---|
| `lm_step.ort` | 30,834,960 | 整条按键路径一张图，权重动态量化为 uint8，已为精简构建的 ONNX Runtime 预先优化 |
| `model.json` | 954 | 运行时除了图之外还需要的东西：缓存形状、上文窗口、BOS 和 UNK 的 id、候选字能否跳过 softmax 读取，以及语料署名 |
| `vocab.json` | 85,613 | token 列表。单个字符就是模型的字；id 是在列表里的位置 |
| `required_operators_and_types.config` | 984 | 图用到的算子和类型，由导出写出（只有注释那一行原本写的是本地构建路径，现在改成了 `lm_step.ort`）：作为 ONNX Runtime `--include_ops_by_config` 和 `--enable_reduced_operator_type_support` 的输入 |
| `LICENSE` | — | Apache-2.0 |
| `NOTICE` | — | 语料署名，必须跟着权重走 |

三个模型文件保持 host 要找的原名：原样放进同一个目录即可。

## 使用

这张图是 [`neural/choice/model.py`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/neural/choice/model.py) 里的 `lm_step`，由 [`neural/choice/export_step.py`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/neural/choice/export_step.py) 导出。权重只存一份，有两种调用方式：

- **文字上屏时**，把上文跑进一个空缓存。上文是以 BOS（`model.json` 里的 `bos`）开头的字 id 序列，最多取最后 `context_chars`（64）个字；不在 `vocab.json` 里的字映射到 `unk`。`context_inputs` 负责构造输入。
- **每次按键时**，只跑缓存里还没有的候选前缀。共同前缀只跑一次，候选的最后一个字从不跑，上文的最后一个节点预测每个候选的第一个字。`step_inputs` 负责构造输入，并返回扩展后的前缀列表供下一次按键使用。

| 输入 | 类型 | 形状 |
|---|---|---|
| `cache_k`, `cache_v` | float32 | [layers, heads, M, head_dim]，M 可以为 0 |
| `cache_h` | float32 | [M, width] |
| `cache_z` | float32 | [M] |
| `nodes`, `positions` | int64 | [N]，所有前缀都已缓存时 N 可以为 0 |
| `visible` | bool | [N, M + N]，每个新节点能看到哪些已缓存节点和新节点 |
| `candidates` | int64 | [B, Lb]，已补齐 |
| `steps` | int64 | [B, Lb]，预测每个候选字的那个节点，索引范围是 M + N 个节点 |
| `lengths` | int64 | [B] |

输出：`scores` [B]，每个候选按字平均的对数概率；以及 `next_k`、`next_v`、`next_h`、`next_z`，即追加了新节点的缓存。下一次按键时把它们喂回去。

两个模型合起来只用 27 个算子，所以可以按两个模型各自的 `required_operators_and_types.config` 把 ONNX Runtime 精简构建到只剩这些（静态链接约 1.5 MB）。`export_step.py --golden` 用 ORT 文件把一个候选池里的用例逐键打一遍，把每次按键的候选和分数写进 `golden.jsonl`，供运行时自己的测试回放。

**权重用 uint8 而不是 int8，是有意的。** ONNX Runtime 按 CPU 特性选择 int8 内核。用 int8 权重时，有和没有 I8MM 的两台 Mac 在 6,317 条用例中有 42 条选了不同的候选；用 uint8 权重，两台给出的分数逐位相同。

## 测量

[`eval/sentences-ctx-v1/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval/sentences-ctx-v1) 是 6,317 条取自训练语料之外文本的用例：2026-09-04 到 10-29 发布的政府文件中的分句，三个众包对话集测试划分里的分句，再加上从每个分句中取一个 2 到 4 字的词，分别以全拼、只打声母和混打三种方式输入。金标准是原作者写下的文字，上文是同一文档里光标之前的文本，候选池录自一个真实的输入法。打分器从同一个池里选中金标准就算对。来源、许可、抽样规则以及与训练语料的重叠检查见该目录的 README；同一批候选池上与其他打分器的比较见 [`docs/measurements.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md)。

| | 有上文（5,468） | 无上文（849） |
|---|---|---|
| **本模型** | **91.4%**（4,997） | 82.7%（702） |
| 本模型 v1（`rerank-v1`） | 91.9% | 82.7% |
| pinyin-ime-keystroke-reranker-8M | 90.1%（4,929） | **83.0%**（705） |
| pinyin-ime-reranker-4M（`model-v1`） | 79.5% | 76.1% |

各行是同一个口径：在同一个候选池里挑按字平均对数概率最高的那个，上文按 MSIME 交给重排器的样子截短（`rerank_context`），每条用例都从空缓存开始，不经过输入法里决定要不要重排的那些规则。按键模型量的就是发布的 `lm_step.ort` 本身（`neural/choice/score_ort.py`），括号里是答对的条数；v1 那一行是 `rerank-v1` 里那份文件的成绩；`model-v1` 那一行就是 [`results.json`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/eval/sentences-ctx-v1/results.json) 里的 `ours-4m`。无上文那一列两个按键模型差 3 条，分不出高低。

### 延迟

在 MSIME 的真实运行时里逐键输入测得（`rerank_latency`，643 条用例取自 sentences-v2 和 typing-chat-v1，约 8,100 次按键，两轮）。整次按键的 p95：

| 设备 | 使用的模型 | 按键 p95 | 同一设备上的 `model-v1` 4.25M |
|---|---|---|---|
| Apple M1 Max / M4 Pro（2 线程） | 8x448 | 约 4 ms | 约 6 ms |
| iPhone 17（2 线程） | 8x448 | 3.8–4.1 ms | 4.8–5.4 ms |
| iPhone XR（2 线程） | 6x256 | 4.4–4.9 ms | 11.1–11.9 ms |
| Pixel 3 XL（4 线程） | 6x256 | 9.3 ms | 38–40 ms |

上表是 v1 时测的。v2 结构没变，延迟也不变：在 M4 Pro 上用 `rerank_latency` 逐键成对测量（单人聊天 test 集，开纠错，4,902 次按键，每个模型两遍），8x448 整句池 9 条时，带模型的按键 p95 v1 是 6.81–6.84 ms，v2 是 6.70–6.78 ms；6x256 整句池 32 条时，v1 是 3.29–3.37 ms，v2 是 3.33 ms（6x256 那次量的是取新权重七成的中间版，结构和权重布局与发布的相同）。

**报分位数，不报均值。** 大多数按键开销很小，均值把贵的那些摊到了便宜的上面。

## 结构

取自 `model.json`：

| | |
|---|---|
| 层数 / 宽度 / 头数 | 8 / 448 / 8（每头维度 56） |
| 位置数 | 256 |
| 上文窗口 | 64 字 |
| 字表 | 12,288 字 |
| 激活函数 | 精确 GELU |
| 候选字的读取方式 | 经过整个字表的 softmax |
| 参数量 | 24.9M |
| 精度 | uint8 权重，动态量化 |

## 语料、许可与署名

**权重是 Apache-2.0**，和训练代码的许可不同（那个仓库是 GPL-3.0，其中 `reference/` 目录是 Apache-2.0）。

`LICENSE` 和 `NOTICE` 放在这个仓库里，而不只是在元数据里写个名字，因为 Apache-2.0 要求再分发者把两者一并交出去，网页上的一个标签不是它们的副本。

训练语料是 [C4](https://huggingface.co/datasets/allenai/c4) 的中文部分（ODC-BY）、[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) 的中文（`cmn_Hani`）部分（ODC-BY）和 [LCCC](https://huggingface.co/datasets/silver/lccc)（MIT），v2 另加了 DeepSeek `deepseek-v4-pro` 为本项目写的合成单人聊天（由 [`neural/teacher/gen_chat.py`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/neural/teacher/gen_chat.py) 生成）。[DeepSeek 使用条款](https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html)把生成内容归用户，并允许用来训练其他模型。**这四类来源都没有相同方式共享（share-alike）的义务**，所以采用这个模型不会把这种义务带到你身上。

这些语料都要求署名跟着用它们训练出来的东西走。署名写在图的元数据里（`attribution`、`license`），ORT 格式会保留它，`model.json` 里也有同一句话，其中合成数据标明了是合成的：

```
Trained on the Chinese portion of C4 (ODC-BY, https://huggingface.co/datasets/allenai/c4), FineWeb-2 (cmn_Hani) (ODC-BY, https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), LCCC (MIT, https://github.com/thu-coai/CDial-GPT) and synthetic one-sided chat messages written by DeepSeek deepseek-v4-pro for this project (neural/teacher/gen_chat.py) (DeepSeek Terms of Use, outputs assigned to the user and usable to train other models, https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html).
```

**再分发时两处都请保留。** 署名是从这次训练实际用到的语料推出来的，不是手写声明的：导出时根据每份下载的语料旁边写下的记录生成它，只要有任何来源是相同方式共享的就拒绝导出。

## 完整性

```
lm_step.ort                           30,834,960 bytes
sha256  827d9ecae4fdfb8ac06d15fbcef827705a55f72ea5e39c96f805f2f1350ee58f

model.json                                   954 bytes
sha256  fdeb4f150fd6a9f937b130ccef4f3248436a9b783202e125fd0a2d1fe3661283

vocab.json                                85,613 bytes
sha256  d7fb2e70de9f0c2c65b822a8fbf604655bfef6d0dcd8f1975d73f1cd5ea7f038

required_operators_and_types.config          984 bytes
sha256  e0c2d4cdbe1c6f1ff9d461a8d3066ca9a4ca42bcfe93ecdc107ad3ec991c22ad
```

加载这些文件的进程能看到每一次按键。**发布之前先校验摘要。** 见 [`SECURITY.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/SECURITY.md)。
