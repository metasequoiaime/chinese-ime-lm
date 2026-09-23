# chinese-ime-lm

<!-- badges:start -->
[![CI](https://img.shields.io/github/actions/workflow/status/metasequoiaime/chinese-ime-lm/ci.yml?branch=main&label=CI)](https://github.com/metasequoiaime/chinese-ime-lm/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/metasequoiaime/chinese-ime-lm/codeql.yml?branch=main&label=CodeQL)](https://github.com/metasequoiaime/chinese-ime-lm/actions/workflows/codeql.yml)
[![License](https://img.shields.io/github/license/metasequoiaime/chinese-ime-lm)](LICENSE)
[![Stars](https://img.shields.io/github/stars/metasequoiaime/chinese-ime-lm?style=flat)](https://github.com/metasequoiaime/chinese-ime-lm/stargazers)
<!-- badges:end -->

面向中文输入法的语言模型工程：语料、统计模型、神经模型，以及判断它们好坏的评测集。

仓库里有两条并行的路线，服务同一件事——在拼音解码给出的若干读法之间做出选择：

| | 路线 | 位置 |
|---|---|---|
| **n-gram** | 语料清洗 → KenLM 统计 → Viterbi 解码 | `corpus/clean/`、`ngram/` |
| **神经模型** | 语料获取 → 字级 Transformer → 候选重排 | `corpus/fetch.py`、`neural/`、`reference/` |

两者用**同一套评测集**（`eval/`）衡量，所以可以直接比较。评测集本身比任何一个模型都更值得先看：它把"某个改动好不好"从意见变成数字，并且把两个会悄悄毁掉测量的陷阱固化成了用例（详见 [`docs/measurements.md`](docs/measurements.md)）。

**与其自己训，不如直接用开源小模型？** 在同一套候选池上量过四个 Qwen：观测上没有一个超过本项目 25.5 MB 的模型，0.6B 与 0.8B 打平 4.5 MB 的那个，而体积差 140 到 380 倍、单次决策慢一个数量级。**但排序质量上的名次差距全部落在噪声里**——52 条上两两分歧只有 2 到 6 条，McNemar 没有一对显著，要检出 1 到 2pp 需要一千到近万条用例。选自训权重的理由是尺寸和延迟这两项确定性测量，不是名次。同一次测量还发现整句集里有 12 条「他/她」在拼音上不可判的用例，本就不该收。全部记在 [`docs/measurements.md`](docs/measurements.md#与开源通用模型比较)。

**当前状态：权重发出来了，但没有进入任何产品。** 神经模型一侧的两份权重作为 [`model-v1`](https://github.com/metasequoiaime/chinese-ime-lm/releases/tag/model-v1) 发布，许可为 Apache-2.0；n-gram 一侧从未有产物出过这个仓库。两条路线都没有进入过 shipped product，而且现有神经模型按音节直接解码的表现比它本想改进的引擎还差。采用之前先把话说在前面。

### 目录

| 路径 | 内容 |
|---|---|
| [`corpus/`](corpus/) | 语料：`fetch.py` 获取可再分发语料，`clean/` 清洗自备语料，`pinyin/` 生成单字拼音表 |
| [`ngram/`](ngram/) | n-gram 路线：KenLM + Viterbi 解码 |
| [`neural/`](neural/) | 神经路线：字级 Transformer 的训练、导出、评测 |
| [`reference/`](reference/) | Rust 推理参考实现，无 unsafe，除 serde 外无依赖（**Apache-2.0**） |
| [`eval/`](eval/) | 25,119 条词级用例 + 60 条整句用例 |
| [`docs/format.md`](docs/format.md) | 模型文件格式规范，用任何语言实现加载器只需要这一篇 |
| [`docs/measurements.md`](docs/measurements.md) | 实测结论：模型在哪里有用、在哪里有害，两个测量陷阱，以及与开源通用模型的对比 |
| [`tests/`](tests/) | 语料清洗的回归测试 |

### 许可按目录区分

仓库整体为 **GPL-3.0**（见 [`LICENSE`](LICENSE)）。

`reference/` 下的 Rust 代码著作权单一，另以 **Apache-2.0** 授权（见 [`reference/LICENSE`](reference/LICENSE)），因为大量开源输入法是 MIT/Apache/BSD，链不了 GPL 代码——第三方可以只取那个目录。

格式规范、评测集和发布的权重都不受代码许可约束。**发布的权重为 Apache-2.0**，理由和 `reference/` 一样：链不了 GPL 的输入法同样用不了 GPL 的权重，而一个没人能采用的公共资源不成其为公共资源。该许可写在权重文件自己的 `__metadata__` 里，所以它不会与权重分离。

权重另有语料的署名义务，见 [`NOTICE`](NOTICE)；那份义务来自语料而非代码，两者不可相互替代。

## 语料来源与许可

**本仓库只分发处理脚本。** 语料和训练好的模型都不入库：`data/`、`model/`、`kenlm_bin/` 下除 `.gitkeep` 外全部被 gitignore，没有任何内容被跟踪。语料由你自己准备。

这一点要紧，因为本项目最初依据的语料**不能自由再授权**，而你下载它的那个仓库上的许可并不能解决这个问题：

- 上游聚合仓库 [brightmart/nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus) 以 MIT 发布。那覆盖的是聚合者自己的工作，不是它打包的第三方文本。
- `wiki2019zh` 派生自中文维基百科，即 **CC BY-SA 3.0**。该许可要求署名，且具传染性：任何实质派生于它的东西都继承这些条款。
- `news2016zh` 是抓取的新闻，`baike2018qa` 是百度百科内容。两者都没有允许再分发的许可。

维基百科下游的任何人都无法把维基的文本重新授权为 MIT，所以聚合仓库的 MIT 徽章只应理解为覆盖其脚本与打包工作。

**这在实践中意味着什么。** 从语料算出来的统计量（n-gram 计数、概率）通常不等于语料本身，但"统计量"和"衍生作品"之间的界线并不清晰——一个能复现源句子的模型，比一张二元频率表更接近衍生作品。所以：

- 本仓库的产物**从未进入过已发布的产品**。[MSIME-Engine](https://github.com/metasequoiaime/MSIME-Engine) 中目前没有任何 n-gram 或 KenLM 模型，也没有加载它们的代码路径。
- **在任何由本管线构建的模型进入输入法之前**，必须先解决所用子集的许可问题：要么把训练限制在可以干净再分发的子集上，要么满足维基派生部分的 CC BY-SA 署名与传染性要求，并把结论记进 Engine 的 `NOTICE.md`，与其他词库来源并列。

`corpus/fetch.py` 正是第一条路线的实现：它只获取许可清晰、可再分发的语料（C4 为 ODC-BY，LCCC 为 MIT），发布的神经模型权重只用这些训练。详见 [`NOTICE`](NOTICE)。

本仓库的 [`LICENSE`](LICENSE)（GPL-3.0）只适用于这里的脚本。它不涉及、也无法授予你喂给这些脚本的语料的任何权利。

## 获取语料

两种方式，按你的用途选：

**自己准备。** 语料来源见上节那个[聚合仓库](https://github.com/brightmart/nlp_chinese_corpus)，解压后放进 `data/` 目录，再用 `corpus/clean/` 清洗。

**让 `corpus/fetch.py` 下载。** 它只取许可清晰的来源，流式下载并直接规范化，原始压缩包不落盘：

```sh
python corpus/fetch.py c4   --out data/c4.txt --max-chars 1_000_000_000
python corpus/fetch.py lccc --out data/lccc.txt --split large
```

## 语料清洗（自备语料）

对于句子中的非中文字符，进行过滤，只要是非中文的字符，一律将其扩充成连通块，然后，将这个连通块当作是一个分隔符，将句子切分成小句子。最后，预处理好的数据就都是一个个由纯粹的中文组成的单独的连通块。

```powershell
python .\corpus\clean\extract_connected_hanzi_components.py
python .\corpus\clean\extract_wiki_connected_hanzi_components.py
python .\corpus\clean\split_all_cleaned_txt_using_space.py
python .\corpus\clean\split_wiki_txt_using_space.py
```

注意，这里是比较耗时的，至少需要半个小时左右，在我的 11 代 intel 处理器上。

## 分词

对上面预处理好的数据进行分词。由于使用的是字级 n-gram，分词只需要把字一个一个分开即可。

## 统计 n-gram

分别计算一元数据和二元数据。

- 一元数据：
  - 纯粹单字在所有字符中出现的频率。
- 二元数据：
  - 按单字切分的 2-gram 数据。
- 三元数据：
  - 按单字切分的 3-gram 数据。

这里的处理就都交给 kenlm 来做，kenlm 的编译和使用参见我的[笔记](https://luflyan.notion.site/VS2026-kenlm-32c722371a17802bac21f6b7e8eebf41?source=copy_link)。需要准备的东西是这种格式的文本语料：

```text
我 爱 北 京
我 爱 中 国
我 爱 上 海
北 京 是 中 国 首 都
```

需要将 [kenlm](https://github.com/kpu/kenlm) 项目编译好的 Release 版本的二进制文件复制到 kenlm_bin 目录下。然后，将生成的 model.arpa 和 model.binary 文件放到相应的 model\all 和 model\only_wiki 目录下。

```powershell
cmd /c ".\kenlm_bin\lmplz.exe -o 3 < .\data\output\all_cleaned_only_wiki_zh_spaced.txt > model.arpa"
.\kenlm_bin\build_binary.exe model.arpa model.binary
mkdir model\only_wiki
mv model.arpa model\only_wiki
mv model.binary model\only_wiki
```

```powershell
cmd /c ".\kenlm_bin\lmplz.exe -o 3 < .\data\output\all_cleaned_spaced.txt > model.arpa"
.\kenlm_bin\build_binary.exe model.arpa model.binary
mkdir model\all
mv model.arpa model\all
mv model.binary model\all
```

## 神经模型

字级 Transformer，在拼音解码拼装出的候选之间重排。它不是普遍有益的——实测见
[`docs/measurements.md`](docs/measurements.md)，结论是：词典对整个键的精确命中已经携带语料词频，
覆盖它们在任何权重下都掉分；而解码器拼装的候选不带词频证据，模型的价值只在那里。

```sh
pip install -r neural/requirements.txt

python corpus/fetch.py c4   --out data/c4.txt --max-chars 1_000_000_000
python corpus/fetch.py lccc --out data/lccc.txt --split large

python neural/train.py --corpus data/c4.txt data/lccc.txt --out runs/desktop --preset desktop
python neural/export.py --run runs/desktop --out dist/model.safetensors --precision int8
```

`corpus/fetch.py` 与 `corpus/clean/` 的关系：后者处理你自己准备好的语料，前者负责**获取**许可清晰、可再分发的语料。两者的中文规范化是同一个算法——保留汉字连通块，其余一律当分隔符。

推理见 `reference/`，格式见 [`docs/format.md`](docs/format.md)。

## 构建与测试

### n-gram 路线

Windows 上用 PowerShell 7.0+：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
```

需要 Python 3.12。

跑解码之前先生成拼音到单字的候选表：

```powershell
python .\corpus\pinyin\make_single_pinyin_table.py
```

然后就可以运行 Viterbi 解码：

```powershell
python .\ngram\viterbi.py
```

语料清洗的回归测试：

```sh
python -m unittest discover -s tests -v
```

### 神经路线

```sh
pip install -r neural/requirements.txt

cd reference
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

参考实现除 serde 外无依赖，测试里手工构造模型文件并加载，因此它同时校验 [`docs/format.md`](docs/format.md) 与实现是否一致。

## 参考资料

- open-gram: <https://github.com/sunpinyin/open-gram>
- nlp_chinese_corpus: <https://github.com/brightmart/nlp_chinese_corpus>

<!-- star-history:start -->
## Star History

<a href="https://star-history.com/#metasequoiaime/chinese-ime-lm&Date">
  <img src="https://api.star-history.com/svg?repos=metasequoiaime/chinese-ime-lm&type=Date" alt="Star History Chart" width="600">
</a>
<!-- star-history:end -->
