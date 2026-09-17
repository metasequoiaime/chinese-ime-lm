# Metasequoia n-gram

<!-- badges:start -->
[![CI](https://img.shields.io/github/actions/workflow/status/metasequoiaime/Metasequoia-n-gram/ci.yml?branch=main&label=CI)](https://github.com/metasequoiaime/Metasequoia-n-gram/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/metasequoiaime/Metasequoia-n-gram/codeql.yml?branch=main&label=CodeQL)](https://github.com/metasequoiaime/Metasequoia-n-gram/actions/workflows/codeql.yml)
[![License](https://img.shields.io/github/license/metasequoiaime/Metasequoia-n-gram)](LICENSE)
[![Stars](https://img.shields.io/github/stars/metasequoiaime/Metasequoia-n-gram?style=flat)](https://github.com/metasequoiaime/Metasequoia-n-gram/stargazers)
<!-- badges:end -->

面向中文输入法的语言模型工程：语料、统计模型、神经模型，以及判断它们好坏的评测集。

仓库里有两条并行的路线，服务同一件事——在拼音解码给出的若干读法之间做出选择：

| | 路线 | 位置 |
|---|---|---|
| **n-gram** | 语料清洗 → 分词 → KenLM 统计 | `preprocessing/`、`test/` |
| **神经模型** | 语料获取 → 字级 Transformer → 导出 safetensors | `corpus/`、`neural/`、`reference/` |

两者用**同一套评测集**（`eval/`）衡量，所以可以直接比较。评测集本身比任何一个模型都更值得先看：它把"某个改动好不好"从意见变成数字，并且把两个会悄悄毁掉测量的陷阱固化成了用例（详见 [`docs/model-overview.md`](docs/model-overview.md)）。

**当前状态：尚未发布任何模型。** n-gram 一侧从未有产物进入过 shipped product；神经模型一侧仍在训练，且现有版本按音节直接解码的表现比它本想改进的引擎还差。发布之前先把话说在前面。

### 目录

| 路径 | 内容 |
|---|---|
| [`preprocessing/`](preprocessing/) | 语料清洗、单字拼音表生成 |
| [`corpus/`](corpus/) | 获取可再分发语料（C4 ODC-BY、LCCC MIT，以及可选的维基与技术文档） |
| [`neural/`](neural/) | 字级 Transformer 的训练、导出、评测 |
| [`reference/`](reference/) | Rust 推理参考实现，无 unsafe，除 serde 外无依赖（**Apache-2.0**） |
| [`eval/`](eval/) | 25,119 条词级用例 + 60 条整句用例 |
| [`docs/format.md`](docs/format.md) | 模型文件格式规范，用任何语言实现加载器只需要这一篇 |
| [`docs/model-overview.md`](docs/model-overview.md) | 神经模型的实测结论：它在哪里有用、在哪里有害 |

### 许可按目录区分

仓库整体为 **GPL-3.0**（见 [`LICENSE`](LICENSE)）。

`reference/` 下的 Rust 代码著作权单一，另以 **Apache-2.0** 授权（见 [`reference/LICENSE`](reference/LICENSE)），因为大量开源输入法是 MIT/Apache/BSD，链不了 GPL 代码——第三方可以只取那个目录。

格式规范、评测集和将来发布的权重都不受代码许可约束；权重的语料署名义务见 [`NOTICE`](NOTICE)。

## Corpus provenance and licensing

This repository distributes **processing scripts only**. No corpus and no trained model is committed: `data/`, `model/` and `kenlm_bin/` are gitignored down to their `.gitkeep`, and nothing under them is tracked. You supply the corpus yourself.

That matters because the corpus this project was built against is not freely relicensable, and the licence on the repository you download it from does not settle the question:

- The upstream aggregator, [brightmart/nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus), is published under MIT. That covers the aggregator's own work, not the third-party text it packages.
- `wiki2019zh` is derived from Chinese Wikipedia, which is **CC BY-SA 3.0**. That licence carries an attribution requirement and is share-alike: anything substantially derived from it inherits those terms.
- `news2016zh` is scraped news articles and `baike2018qa` is Baidu Baike content. Neither carries a licence that permits redistribution.

Nobody downstream of Wikipedia can relicense Wikipedia's text under MIT, so treat the aggregator's MIT badge as covering its scripts and packaging only.

**What this means in practice.** Statistics computed from a corpus (n-gram counts, probabilities) are generally not the corpus, and the boundary between "statistics" and "a derivative work" is not sharp — a model that can reproduce source sentences is much closer to a derivative than a table of bigram frequencies. So:

- Nothing produced here has entered the shipped product. There is currently no n-gram or KenLM model in [MSIME-Engine](https://github.com/metasequoiaime/MSIME-Engine) and no code path that loads one.
- **Before any model built from this pipeline ships inside the input method**, the licensing of the specific subsets used has to be settled first: either restrict training to subsets that are cleanly licensed for redistribution, or satisfy CC BY-SA attribution and share-alike for the Wikipedia-derived portion, and record the outcome in the Engine's `NOTICE.md` alongside the other dictionary sources.

The `LICENSE` in this repository (GPL-3.0) applies to the scripts here. It says nothing about, and cannot grant any rights to, the corpus you feed them.

## Corpus collection

这里使用的语料来源是这个[仓库](https://github.com/brightmart/nlp_chinese_corpus)。将其解压后放到 data 目录下。

## Data preprocessing

对于句子中的非中文字符，进行过滤，只要是非中文的字符，一律将其扩充成连通块，然后，将这个连通块当作是一个分隔符，将句子切分成小句子。最后，预处理好的数据就都是一个个由纯粹的中文组成的单独的连通块。

```powershell
python .\preprocessing\generate_cleaned_txt\extract_connected_hanzi_components.py
python .\preprocessing\generate_cleaned_txt\extract_wiki_connected_hanzi_components.py
python .\preprocessing\generate_cleaned_txt\split_all_cleaned_txt_using_space.py
python .\preprocessing\generate_cleaned_txt\split_wiki_txt_using_space.py
```

注意，这里是比较耗时的，至少需要半个小时左右，在我的 11 代 intel 处理器上。

## Segmentation

对上面的预处理好的数据进行分词。由于使用的是字级的 n-gram，所以，分词只需要将一个一个字分开即可。

## N-gram info counting

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
[`docs/model-overview.md`](docs/model-overview.md)，结论是：词典对整个键的精确命中已经携带语料词频，
覆盖它们在任何权重下都掉分；而解码器拼装的候选不带词频证据，模型的价值只在那里。

```sh
pip install -r neural/requirements.txt

python corpus/corpus.py c4   --out data/c4.txt --max-chars 1_000_000_000
python corpus/corpus.py lccc --out data/lccc.txt --split large

python neural/train.py --corpus data/c4.txt data/lccc.txt --out runs/desktop --preset desktop
python neural/export.py --run runs/desktop --out dist/model.safetensors --precision int8
```

`corpus/corpus.py` 与 `preprocessing/` 的关系：后者处理你自己准备好的语料，前者负责**获取**
许可清晰、可再分发的语料。两者的中文规范化是同一个算法——保留汉字连通块，其余一律当分隔符。

推理见 `reference/`，格式见 [`docs/format.md`](docs/format.md)。

## How to build and run tests

For windows, you can use PowerShell 7.0+ like this:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install regex
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install opencc
```

注意，这里要用 Python 3.12。

跑测试代码之前，需要先生成一下拼音到单个汉字的候选项列表的字典文件，

```powershell
python .\preprocessing\generate_single_hanzi_pinyin_tbl\make_single_pinyin_table.py
```

然后，就可以运行测试了，

```powershell
python .\test\viterbi_no_pruning.py
```

## Reference

- open-gram: <https://github.com/sunpinyin/open-gram>
- nlp_chinese_corpus: <https://github.com/brightmart/nlp_chinese_corpus>

<!-- star-history:start -->
## Star History

<a href="https://star-history.com/#metasequoiaime/Metasequoia-n-gram&Date">
  <img src="https://api.star-history.com/svg?repos=metasequoiaime/Metasequoia-n-gram&type=Date" alt="Star History Chart" width="600">
</a>
<!-- star-history:end -->
