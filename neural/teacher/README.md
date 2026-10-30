# 蒸馏数据与基线（2.0 第一步）

2.0 的打算是从一个大语言模型老师蒸馏进现在的学生结构（`neural/model.py`，4.25M 与 24.9M 两档），引擎不用改。这个目录是第一步：按 [`eval/sentences-ctx-v1/`](../../eval/sentences-ctx-v1/)（`eval/large-context-set` 分支）的口径从两份训练语料里抽用例，录下 MSIME 给的候选，切成和评测集同样形状的候选池，并在留出的 dev 上量现有两个模型和引擎原序的命中。**这一步不训练任何东西**；老师的打分与学生的训练在后面。

| 文件 | 作用 |
|---|---|
| `requirements.txt` | 老师管线的依赖（`mlx-lm==0.32.2`），这一步还没用到 |
| `build_cases.py` | 从 C4 与 LCCC 的原始文件抽分句、标拼音、生成打法，分 train / dev，排除与评测集重合的 |
| `record.py` | 分块跑 MSIME `convert_eval --dump`、切池、统计覆盖 |
| `score_dev.py` | 在 dev 池上跑 `reference/examples/pool.rs`，按子集报命中与 Wilson 95% 区间 |
| `v1/` | 这一版的 `build.json`（抽样记录）、`*-coverage.json`（录候选记录）、`dev-results.json`（基线）、`files.json`（大文件的哈希） |

大文件不进仓库，放在 `~/worktrees/_lm-teacher-data/`（`cases/`、`pools/`、`baselines/`），哈希在 `v1/files.json`。

## 为什么不直接读 `data/c4.txt`、`data/lccc.txt`

这两个文件是 `corpus/fetch.py` 写的：只留连续汉字段，标点、数字、字母都当切分边界扔掉；全语料按行去重；没有文档边界。所以文件里一个分句的上一行不一定是它前面的文字（中间可能被去重删掉了几行，也可能是上一篇文档的末尾），而评测集的上文带标点、是光标前的原文。

所以读的是这两个文件的来源：C4 中文的原始分片（`c4-zh.tfrecord-00000/00001-of-01024.json.gz`，`data/c4.txt` 就是从分片 0 起建的）和 `data/raw/lccc_large.jsonl.gz`（`data/lccc.txt` 的来源）。过滤与 `fetch.py` 相同：繁体文档 / 轮次不要（`is_traditional_document`），C4 不是正文的文档不要（`is_prose`），带繁体标记字的分句不要（与 `segment` 相同）。

**这意味着 dev 里的文字在现有两个模型的训练语料里**：dev 的 1842 个进池分句，每一个都是 `data/c4.txt` 或 `data/lccc.txt` 里的一整行（逐行比对过）。dev 对蒸馏数据是留出的，对现有模型不是。下面的基线因此偏乐观，不能和 sentences-ctx-v1 上的数字直接比；那套集的文字不在任何一方的训练语料里，仍然是最后的裁判。

## 抽样

与 sentences-ctx-v1 相同的部分（函数照抄 `build_set.py`，标注了出处）：

- **分句**：两侧是标点（`，。！？；：、` 等）、空格或文本边界的 4 到 16 个连续汉字。
- **上文**：同一篇文档或同一轮对话里分句之前的全部文字，保留标点，最多最后 1024 字节。C4 文档的行与行之间什么也不加（回车不是上屏文字）；LCCC 只用同一轮，不用对方说的话。
- **读音**：pypinyin 0.55.0 按整段读，一字一音节；读成全拼打不出的音（`n`、`m` 等）的分句不收。
- **打法**：每个分句全拼打一次；分句里用 jieba 0.42.1 取一个 2 到 4 字的词（或相邻两词），全拼、只打声母、混打各一次。随机数的抽取顺序和 `typed_cases` 一样。

不同的地方：

- **分句的声母、混打两种打法不录**。MSIME 不从缩写按键拼整句，评测集录它们只为统计这一点，不成池。所以每个分句 4 条用例（没有可取的词时 1 条），不是 6 条。
- **C4 的段落**只有含 `，。；：！？` 之一时才出分句（评测集对政府文件的规则），挡掉导航、菜单、标题。上文仍然包括这些行。
- **分层**：和评测集的对话集一样，单元（C4 的文档、LCCC 的一轮）打乱后交替分给「无上文」「有上文」两层；无上文层取单元开头的分句，有上文层随机取不在开头的。LCCC 一段对话至多 3 条，C4 一篇文档有上文的至多 2 条。C4 文档很少以分句开头（分片 0 大约九篇里一篇），所以无上文层主要来自 LCCC——评测集的无上文用例也全是对话。

| | C4 无上文 | C4 有上文 | LCCC 无上文 | LCCC 有上文 | 分句 | 用例 |
|---|---|---|---|---|---|---|
| train | 3000 | 35000 | 47000 | 15000 | 100000 | 399664 |
| dev | 100 | 1050 | 1400 | 450 | 3000 | 11997 |

每个单元按其编号的哈希归 train 或 dev（25 个里 1 个归 dev），先抽 dev。LCCC 从 1200 万段对话里随机读 60 万段；C4 两个分片过滤后剩 71090 篇。种子固定（`teacher-20261030`），同样的输入重跑，四个输出文件逐字节相同。

**排除**：分句的金标准与任一评测集（`eval/sentences-v1.tsv`、`eval/quanpin-words-v1.tsv`、`eval/open-models/pool.jsonl`、`eval/sentences-ctx-v1/sampled.jsonl.gz`）的任一金标准相同，或者它的金标准或上文与那些用例的金标准或上文共有一段 8 个连续汉字（与评测集重合检查的 `window8` 同长），就不要；train 再对 dev 做一遍同样的检查；金标准在两份里都不重复。被挡掉的数目在 `v1/build.json`：

| | 金标准相同 | 金标准共有 8 字 | 上文共有 8 字 | 重复 |
|---|---|---|---|---|
| dev C4 | 22 | 4 | 179 | 64 |
| dev LCCC | 5 | 0 | 0 | 9 |
| train C4 | 4204 | 1019 | 16091 | 3625 |
| train LCCC | 844 | 77 | 24 | 2047 |

train C4 被上文挡掉的多，是因为 C4 里大量页面共用同一套模板文字（博彩、SEO 站），和 dev 的 C4 上文撞上。词用例的金标准是 2 到 4 字的常用词，不和评测集的词按相同来排除，否则要扔掉一大片常用词；它的上文就是分句的上文加分句前半，已经在分句那一级查过。

## 录候选

MSIME `c20b3f447654401b3cc4a012ca309b735f8105c2`（develop，2026-10-30），词库 dict-v2.0.41（`resources/desktop-dictionary.lock.json`，词库源 `09a69f20b4726f75d4aa318975e3e891c1135aaf`），资源目录与锁里去掉 `sentence-model.safetensors`，所以是引擎自己的顺序。与 sentences-ctx-v1 用的 `e6b494be61` 之间只有 Android 的按键音改动，引擎相同。`convert_eval` 取自停在这个提交、没有改动的 MSIME worktree 的 `target/release/examples/`（本次没有重新编译），SHA-256 在 `v1/*-coverage.json`。

切池规则与 sentences-ctx-v1 相同：全拼分句取 `generated`、`fallback` 两个来源、与金标准等长的候选；词取与金标准等长的前 9 个不同候选、不论来源；金标准不在池里或池里只有一个的丢掉。

速度：本机 4 个 `convert_eval` 并行，train 的 399664 条用了 487 秒（约 820 条每秒），dev 的 11997 条 35 秒，所以没有缩减规模。每条 dump 约 40 KB，`record.py` 每 4000 条一块、切完即删。

| 子集 | train 用例 | 金标准在候选里 | 进池 | 引擎首选已对 | dev 用例 | dev 进池 |
|---|---|---|---|---|---|---|
| 分句，全拼，有上文 | 50000 | 32878 | 30352 | 21247 | 1500 | 940 |
| 分句，全拼，无上文 | 50000 | 32712 | 30560 | 21946 | 1500 | 902 |
| 词，全拼，有上文 | 84705 | 78482 | 78333 | 65332 | 2503 | 2327 |
| 词，全拼，无上文 | 15183 | 14048 | 14028 | 12085 | 496 | 456 |
| 词，只打声母，有上文 | 84705 | 54489 | 33247（另 491 条只有 1 个） | 13312 | 2503 | 971 |
| 词，只打声母，无上文 | 15183 | 9673 | 6792（另 98） | 3119 | 496 | 237 |
| 词，混打，有上文 | 84705 | 54402 | 43513（另 6598） | 25396 | 2503 | 1320 |
| 词，混打，无上文 | 15183 | 9606 | 7651（另 1390） | 4935 | 496 | 268 |

train 共 244476 个池，dev 7421 个。池的每行：`id`、`split`、`level`、`style`、`source`、`unit`、`input`、`gold`、`syllables`、`readings`、`context`、`candidates`、`sources`，词还有 `clause`。行是按种子打乱过的，所以任何前缀都是随机子样本，老师打分可以只取一段。

## dev 基线

`reference/examples/pool.rs`（与 sentences-ctx-v1 分支上的同一份：上文先按 MSIME 的 `rerank_context` 截）挑每字平均对数概率最高的候选；引擎原序取第一个。只报命中，不报耗时。模型是 `dist/` 里发布的两个（`sentence-model.safetensors` `86ac5295…`，`sentence-model-desktop.safetensors` `0a6ecba6…`）。

| 子集 | 用例 | 引擎原序 | 4.25M | 24.9M |
|---|---|---|---|---|
| 分句，全拼 | 1842 | 1298（70.5%，68.3–72.5） | 1475（80.1%，78.2–81.8） | 1603（87.0%，85.4–88.5） |
| 　有上文 | 940 | 639（68.0%，64.9–70.9） | 724（77.0%，74.2–79.6） | 800（85.1%，82.7–87.2） |
| 　无上文 | 902 | 659（73.1%，70.1–75.8） | 751（83.3%，80.7–85.5） | 803（89.0%，86.8–90.9） |
| 　C4 有上文 | 629 | 417（66.3%，62.5–69.9） | 468（74.4%，70.9–77.7） | 531（84.4%，81.4–87.0） |
| 　C4 无上文 | 55 | 33（60.0%，46.8–71.9） | 42（76.4%，63.6–85.6） | 46（83.6%，71.7–91.1） |
| 　LCCC 有上文 | 311 | 222（71.4%，66.1–76.1） | 256（82.3%，77.7–86.2） | 269（86.5%，82.2–89.8） |
| 　LCCC 无上文 | 847 | 626（73.9%，70.9–76.8） | 709（83.7%，81.1–86.0） | 757（89.4%，87.1–91.3） |
| 词，全拼 | 2783 | 2336（83.9%，82.5–85.3） | 2352（84.5%，83.1–85.8） | 2448（88.0%，86.7–89.1） |
| 　有上文 | 2327 | 1953（83.9%，82.4–85.4） | 1967（84.5%，83.0–85.9） | 2058（88.4%，87.1–89.7） |
| 　无上文 | 456 | 383（84.0%，80.3–87.1） | 385（84.4%，80.8–87.5） | 390（85.5%，82.0–88.5） |
| 词，只打声母 | 1208 | 531（44.0%，41.2–46.8） | 777（64.3%，61.6–67.0） | 857（70.9%，68.3–73.4） |
| 　有上文 | 971 | 413（42.5%，39.5–45.7） | 628（64.7%，61.6–67.6） | 703（72.4%，69.5–75.1） |
| 　无上文 | 237 | 118（49.8%，43.5–56.1） | 149（62.9%，56.6–68.8） | 154（65.0%，58.7–70.8） |
| 词，混打 | 1588 | 981（61.8%，59.4–64.1） | 1178（74.2%，72.0–76.3） | 1251（78.8%，76.7–80.7） |
| 　有上文 | 1320 | 804（60.9%，58.2–63.5） | 972（73.6%，71.2–75.9） | 1047（79.3%，77.0–81.4） |
| 　无上文 | 268 | 177（66.0%，60.2–71.5） | 206（76.9%，71.5–81.5） | 204（76.1%，70.7–80.8） |

括号里是命中率和 Wilson 95% 区间。MSIME 自己不重排词典行，词那几行量的是模型，不是 MSIME 的行为。再说一遍：这些文字在两个模型的训练语料里。

## 要注意的

- **C4 的垃圾页**：train 的 38000 个 C4 分句里有 5607 个（14.8%）的上文或金标准带博彩类词（彩票、时时彩、快三、11选5、开奖、棋牌等，LCCC 是 62000 里 13 个）。`fetch.py` 的行级去重在这里不起作用，因为抽的是原始文档。这一步没有过滤；老师打分前要不要按词表去掉，留给下一步决定。
- 排除检查依赖 sentences-ctx-v1 的 `sampled.jsonl.gz`（SHA-256 `00687bff…`），它现在只在 `eval/large-context-set` 分支上。那份文件如果再变，要重跑 `build_cases.py`。
- `reference/examples/pool.rs` 取的是 `eval/large-context-set` 分支的版本，两边内容逐字节相同，合并时不冲突。
- pypinyin 0.55 遇到 CJK 兼容汉字（U+F900 区，如 U+F98E 年）时 `errors` 回调返回 `None` 会抛 `TypeError`；`build_cases.py` 只在这种段落改用占位符重读，其余段落的读法与 `build_set.py` 完全相同。

## 复现

```sh
# 语料：C4 中文分片 0 和 1（https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-zh.tfrecord-0000{0,1}-of-01024.json.gz，
# SHA-256 见 v1/build.json），以及 corpus/fetch.py lccc --split large 下载的 data/raw/lccc_large.jsonl.gz
python neural/teacher/build_cases.py --c4 <shard0> --c4 <shard1> --lccc data/raw/lccc_large.jsonl.gz \
  --eval-ctx <eval/sentences-ctx-v1/sampled.jsonl.gz> --out-dir <data>/cases

# MSIME c20b3f4476 的源码树里编译 convert_eval；资源目录按 resources/desktop-dictionary.lock.json 取齐、去掉 sentence-model.safetensors，锁里也去掉这一项
cargo build --release -p msime-input-runtime --example convert_eval
for split in dev train; do
  python neural/teacher/record.py --convert-eval <convert_eval> --resources <res> --resource-lock <lock-no-model.json> \
    --msime-commit c20b3f447654401b3cc4a012ca309b735f8105c2 --cases <data>/cases/$split.jsonl --tsv <data>/cases/$split.tsv \
    --work <data>/work/$split --out <data>/pools/$split.jsonl --coverage <data>/pools/$split-coverage.json --jobs 4
done

cargo build --release --manifest-path reference/Cargo.toml --example pool
python neural/teacher/score_dev.py --pool <data>/pools/dev.jsonl --pool-bin reference/target/release/examples/pool \
  --model ours-4m=dist/sentence-model.safetensors --model ours-25m=dist/sentence-model-desktop.safetensors \
  --picks-dir <data>/baselines --out <data>/baselines/dev-results.json
```

`build_cases.py` 需要 pypinyin 0.55.0 和 jieba 0.42.1（与 sentences-ctx-v1 相同的版本），都不是本仓库的依赖。
