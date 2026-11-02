# 带上文的整句与选词评测集（sentences-ctx-v1）

[`docs/measurements.md`](../../docs/measurements.md#与开源通用模型比较) 那张 52 条的比较表回答不了 1 到 3pp 的差别，而且 52 条里没有一条带上文。这个目录是它的扩大版：从训练语料之外的文字里抽出 2485 个分句，录下 MSIME 当前引擎给的候选，切出 1631 条整句用例（1196 条带上文）和 4686 条选词用例（全拼、只打声母、全拼和声母混打三种打法），再让同一批打分器各挑一次。

和 [`eval/open-models/`](../open-models/) 一样，**这里只量「在同一个池里挑一个」这一项能力**，不代表各家模型在自己输入法里的端到端表现。逐例选择、区间、配对检验在 `results.json`，结论写在 `docs/measurements.md`。

与 MSIME 仓库里那份收割集 `resources/eval/sentences-v2.tsv` 无关。

## 文件

| 文件 | 内容 |
|---|---|
| `build_set.py` | 下载三类来源（`fetch`），切分句、标拼音、抽样、生成三种打法（`build`） |
| `sampled.jsonl.gz` | 抽出的全部 14868 条打字用例（2485 个分句和从其中 2471 个里取的一个词，各三种打法），带来源、读音和上文 |
| `sources.json` | 每份来源的地址、提交号、文件 SHA-256、取得日期；政府文件逐篇列出发文号、发文机关、发布日期和附件 |
| `review.tsv` | 人工读过之后去掉的 11 个分句和理由 |
| `check_overlap.py`、`overlap.json` | 与各打分器训练语料的重合检查和结果 |
| `build_pool.py` | 从 MSIME `convert_eval --dump` 切出候选池，统计落在池外的用例 |
| `pool.jsonl` | 6317 条用例的候选池，引擎原序 |
| `pool-sentences-v1.jsonl` | 本仓库 `eval/sentences-v1.tsv` 那 60 条用同一个引擎重录的池（48 条） |
| `coverage.json` | 每个子集抽了多少、金标准出现在候选里多少、进池多少、引擎首选对多少 |
| `run_scorers.sh` | 依次跑全部打分器；`TIMING_ONLY=1` 只在 `pool-timing.jsonl` 上跑计时的那一轮（默认 5 轮） |
| `timing_pool.py`、`pool-timing.jsonl` | 计时用的固定子集：八层（整句或词 × 打法 × 有没有上文）各 60 条，共 480 条，种子固定 |
| `models.json` | 每个打分器的来源、哈希、运行方式，以及各自的输入怎么喂 |
| `summarize.py` | 汇成 `results.json`、`results.csv`、`picks.json` |
| `results.json` | 每个打分器在每个子集上的命中、Wilson 95% 区间；每个子集里任意两个打分器的 McNemar 精确检验；每个子集能分辨的净差；每个子集里被计时的用例（`timed_n`）一次决策的 p50、p95。耗时只来自另跑的计时那一轮（`TIMING_ONLY=1`，`summarize.py --timing-dir`），录选择那一轮和别的构建、测试共用一台机器，它的耗时不用；计时那一轮的机器、负载和每一段的时间在 `timing` |
| `picks.json` | 每个打分器在每条用例上选了第几个 |

## 来源与许可

**选来源的约束**：金标准不能取自任何被比较打分器的训练语料。本项目的模型用中文 C4 和 LCCC 训练（`data/*.source.json`、`runs/*/config.json`）；MSIME 的 n-gram 表来自中文维基百科 20260901 的第一个分卷（MSIME `resources/dictionary-sources.lock.json`），词库来自雾凇拼音；知意的训练文本有 SkyPile、LCCC、C4、SODA、Tatoeba、FineWeb 和 FineWeb-2（知意 README）。所以 C4、LCCC、维基百科、Tatoeba 都不用，网页语料一概不用。

| 来源 | 内容 | 许可 | 分句 | 进池（整句） |
|---|---|---|---|---|
| `gov` | 中国政府网政策文件库里国务院、国务院办公厅和各部委 2026-09-04 至 2026-10-29 发布的 55 份文件（14 份国务院、41 份部门），连同 24 个 PDF 附件 | 不受著作权法保护：《著作权法》第五条，「法律、法规，国家机关的决议、决定、命令和其他具有立法、行政、司法性质的文件，及其官方正式译文」 | 1035 | 735 |
| `kdconv` | [KdConv](https://github.com/thu-coai/KdConv) `653db76` 测试集，众包的电影、音乐、旅游闲聊 | Apache-2.0 | 583 | 360 |
| `crosswoz` | [CrossWOZ](https://github.com/thu-coai/CrossWOZ) `df82c9f` 测试集，众包的北京旅游任务对话 | Apache-2.0 | 433 | 245 |
| `risawoz` | [RiSAWOZ](https://github.com/terryqj0107/RiSAWOZ) `1e2d3be` 测试集，众包的 12 个领域任务对话 | MIT | 434 | 291 |

政府文件全部发布于 MSIME 的维基分卷（2026-09-01）之后，也晚于 C4（2020 年的抓取）、SkyPile（2023-12）、FineWeb（最新一批是 CC-MAIN-2025-26）和 FineWeb-2（最后更新 2025-10-27）的抓取，以及四个 Qwen 的发布（最晚的 Qwen3.5 是 2026-02-28），所以不可能在其中任何一个里。三个对话集是 2020 年发布在 GitHub 上的 JSON，不是网页文字，LCCC-large 的组成（CDial-GPT README 列出的十个来源）里没有它们；它们有没有经由别处转载进网页语料，这里查不了（见下）。

再分发：政府文件不受著作权保护；KdConv、CrossWOZ 的 Apache-2.0 要求保留许可与归属，RiSAWOZ 的 MIT 要求保留版权声明——本文件就是归属声明：`sampled.jsonl.gz` 与 `pool.jsonl` 里的对话文字出自上面三个仓库，版权归其作者，按各自许可（Apache-2.0：<https://www.apache.org/licenses/LICENSE-2.0>；MIT：RiSAWOZ 仓库的 `LICENSE`）再分发，本仓库的改动是切分句、标拼音和抽样。每条用例的 `unit` 字段指回原文（政府文件是 `content_<编号>`，在 `sources.json` 里有地址；对话是对话编号和轮次）。

## 抽样

**分句**：一段原文里两侧都是标点（`，。！？；：、` 等）、空格或段落边界的一段连续汉字，4 到 16 个字。旁边紧挨着数字、字母、引号、括号的不算，因为那是一个更长的、不全用拼音打的东西的一部分。这是用户一口气打完再上屏的那一段。

**上文**就是输入法看得到的光标前的文字，同一篇文字里这个分句之前的全部内容，保留标点，最多留最后 1024 字节（MSIME 保留的已上屏文字就是这么多，没有一个打分器读得更远）：

- 政府文件：标题、正文、附件依次接起来，段与段之间什么也不加——回车不是输入法上屏的文字。所以政府文件的分句几乎都有很长的上文（中位数 342 字，被 1024 字节截住）。
- 对话：只有同一轮里这个分句之前的部分（中位数 8 到 14 字）。对方说的话在屏幕上，但不是用户上屏的文字，输入法看不到，MSIME 的收割集已经量过把对方的话当上文没有收益（`resources/eval/README.md`）。所以对话的「无上文」是一轮的第一个分句，和原来那 60 条手写句同一种情形。

**抽法**：每份政府文件随机取至多 20 个分句，已经从前面的文件取过的不再取（这些文件共享大量套话）。对话集把所有轮次打乱后交替分给两层：「无上文」层取一轮的第一个分句（要求它就在轮首），「有上文」层随机取一轮里不在轮首的一个分句；每层各有定额（KdConv 300/300，CrossWOZ 和 RiSAWOZ 各 250/220），一轮最多出一条，一段对话最多出三条。全集按金标准去重。随机数按来源各用一个种子（`20261030-<来源>`），改一个来源的规则不动别的来源。

**拼音**用 pypinyin 按整段读（多音字看上下文），一个字一个音节，不加隔音符（与原 60 条一致，`fangan` 不写成 `fang'an`）；读成 `n`、`m`、`hm` 这类全拼里没有的音节的分句不收。

**选词用例**：每个分句里随机取一个 2 到 4 字的词或相邻两个词（jieba 切词），上文是分句的上文加上分句里这个词之前的部分。切不出这样一个词的 14 个分句不出选词用例，所以词是 2471 个。

**三种打法**：全拼；只打声母，每个音节打第一个字母（中国人 `zgr`）；混打，随机让一部分音节全打、其余只打第一个字母，两种至少各一个（我是一个 `wsyige`）。分句和词都这样打三遍。

**人工复查**：引擎录完候选之后、任何打分器运行之前，读过全部进池的对话分句，去掉了金标准本身不是规范写法的 11 条（`review.tsv`：错字、的/得混用、指物用「他」、网络谐音）；政府文件抽读了一遍，没有发现要去掉的。复查没有看任何打分器的输出。

## 候选池

用 MSIME develop `e6b494be61`（2026-10-30）和它锁定的词库 dict-v2.0.41 录候选，资源目录里拿掉两个句子模型，所以是引擎自己的顺序；每条用例先清空再写入它的上文（`convert_eval` 的第 6 列），再打字：

```sh
# MSIME e6b494be61 的源码树里
cargo build --release -p msime-input-runtime --example convert_eval
target/release/examples/convert_eval --resources <按锁取齐、去掉两个模型的资源目录> \
  --resource-lock <去掉 sentence-model.safetensors 那一项的锁> --set sampled.tsv --dump dump.jsonl
python eval/sentences-ctx-v1/build_pool.py --dump dump.jsonl --sampled sampled.jsonl --overlap eval/sentences-ctx-v1/overlap.json \
  --out eval/sentences-ctx-v1/pool.jsonl --stripped pool-stripped.jsonl --coverage eval/sentences-ctx-v1/coverage.json
```

`sampled.tsv` 由 `build_set.py build` 写出（`sampled.jsonl.gz` 解压后就是它的 JSON 版）。dump 有 680 MB，不收进仓库。

**为什么换成当前的引擎**：原表固定在 2026-09-19 的构建，是为了逐条复现已经发布的那张表。这套用例没有要复现的旧数字；要回答的是打分器在当前产品给的候选上挑得怎样，所以用当前的引擎和词库。换引擎本身的影响用原 60 条量过：同一份 `eval/sentences-v1.tsv` 在新构建上进池的从 52 条变成 48 条（少了 s-024、s-036、s-040、s-052、s-060，多了 s-035），池子从每条 6 到 7 个候选变成每条正好 6 个，来源全是 `generated`（新引擎不再给 `fallback`），引擎原序从 41/52 变成 43/48。各打分器在这 48 条上的结果在 `results.json` 的 `sentences_v1_on_develop`。

**切池规则**：

- 全拼的分句：与 `eval/open-models/build_pool.py` 完全相同，取 `generated`、`fallback` 两个来源、与金标准等长的候选，引擎顺序。
- 词：MSIME 对这些输入给的是词典行，不是解码器拼装的，所以不能按解码器来源取。取与金标准等长的前 9 个不同候选，不论来源，引擎顺序，至少 2 个。9 是 MSIME 重排器一次比较的条数（一页），也是青简默认档位交给知意的条数；知意输入法比两页。等长只用到金标准的字数，运行时从按键也知道：一个全拼音节或一个声母就是一个字。
- 只打声母或混打的**分句不成池**：MSIME 不从缩写的按键拼整句。2485 个分句只打声母时，没有一个分句的候选里有解码器拼出的句子（来源 `generated`、`fallback`）；混打时有 9 个，其中 4 个与金标准等长。金标准出现在候选里的，打声母 124 个、混打 128 个，全是词典行（来源 `database`），其中 81 个、83 个是四个字（成语、固定说法），没有一个超过 8 个字。所以 `clause/initials`、`clause/mixed` 两个子集是空的，这是对引擎的测量，不是抽样的缺口（数字由 dump 和 `sampled.jsonl` 按 `id` 对出来）。

金标准不在池里的用例丢掉。**落在池外的本身是对引擎候选生成的测量**（`coverage.json`）：

| 子集 | 抽样 | 金标准出现在候选里 | 进池 | 引擎首选已对 |
|---|---|---|---|---|
| 分句，全拼，有上文 | 1739 | 1298 | 1196（68.8%） | 838 |
| 分句，全拼，无上文 | 746 | 467 | 435（58.3%） | 293 |
| 词，全拼 | 2471 | 2358 | 2356 | 1969 |
| 词，只打声母 | 2471 | 1623 | 1043（另 33 条池里只有 1 个） | 426 |
| 词，混打 | 2471 | 1614 | 1287（另 263 条池里只有 1 个） | 760 |

分句里有 854 条没进池：720 条金标准根本不在候选里，134 条在候选里但不是解码器拼出的等长整句（是词典行或别的长度）。读音标错也会让金标准出不来：拿 MSIME 自己的读音（`typing_sim --lint-readings`）对一遍，854 条里能确认是 pypinyin 读错的 22 条左右（「多长时间」「多长」读成 duozhang 13 条，长效、全长、流行乐、传统工艺、什刹海等），其余不一致是引擎那边的读音（有 读成 wei、呀 读成 xia）或两读皆可（谁 shui/shei）。所以落在池外的绝大部分是候选生成本身没给出来。

## 重合检查

`check_overlap.py` 把每份语料切成连续汉字段（本项目的训练文件本来就是这样存的；Tatoeba 先繁转简，维基分卷按 `<text>` 切），对 2485 个分句查三件事：整个分句是不是语料里的一整段（`exact`）；分句里有没有任意 8 个字的片段出现在语料里（`window8`，MSIME 给打字集做过同样的检查）；分句和它前面那一段是不是在语料里也前后相连（`document`，原文整篇在语料里时就是这个样子）。维基还用繁体形式再查一遍。

| 语料 | 字数 | exact | 其中 ≥8 字 | window8 | document |
|---|---|---|---|---|---|
| 本项目 C4（`data/c4.txt`） | 2.47 亿 | 319 | 43 | 222 | 7 |
| 本项目 LCCC（`data/lccc.txt`） | 2.47 亿 | 197 | 5 | 49 | 0 |
| Tatoeba 中文（2026-10-24 导出） | 92 万 | 18 | 0 | 6 | 0 |
| zhwiki 20260901 第一分卷（MSIME 的 n-gram 语料） | 1.33 亿 | 137 | 11 | 44 | 11 |

（2485 个分句里 1575 个不短于 8 个字。）`document` 的 18 次全是政府文件的套话：主送机关的排列（「各省、自治区、直辖市人民政府，国务院各部委」）、「构成犯罪的，依法追究刑事责任」、省份名称的罗列——早年的文件在网上和维基上一字不差地出现过，不是这次的文件本身。整句一样的短分句是常用说法（「电话号码是多少」）。进池的整句用例里在任何一份语料里整句出现过的标在 `seen_in`（269 条），`results.json` 另报去掉它们之后的子集 `clause/full/unseen`。

**没有查的**：SkyPile、FineWeb、FineWeb-2 没有本地副本，Hugging Face 的检索接口只索引 FineWeb-2 中文部分（1.6 TB）开头的几 GB，查了也说明不了什么；Qwen 的训练语料不公开；青简含章两款（2026-10-21 发布）的训练语料没有找到说明，所以政府文件对含章两款只能说「发布日期很近，在不在它们的语料里不知道」。三个对话集只能说「不在本地能查的语料里」，它们 2020 年起就在 GitHub 上，被某个大模型的训练语料收进去是可能的。

## 子集与打分器能吃什么

`run_scorers.sh` 让每个打分器跑三份池：`pool.jsonl`；`pool-stripped.jsonl`（有上文的用例把上文拿掉，池不变，量上文对它起了什么作用）；`pool-sentences-v1.jsonl`。Qwen 的三种起始 token（`docs/measurements.md`「复测」一节）只有 `eos` 跑全部三份池，它在 52 条上四个模型都是三种里最好的；`none`、`endoftext` 只跑 48 条的 `pool-sentences-v1.jsonl`，接上原表。三种都跑大池要多两倍的时间，换来的只是两种已知更差的测法（`QWEN_POOL_STARTS="none endoftext eos"` 可以全跑）。各打分器拿到什么，照它自己的输入法（详细见 `models.json` 的 `fed`）：

| 打分器 | 读上文 | 读按键 | 全拼分句 | 全拼词 | 声母、混打的词 |
|---|---|---|---|---|---|
| 本项目两个模型 | MSIME 交给重排器的样子（`rerank_context`） | 否 | 照 MSIME | 模型能打分；MSIME 自己不重排词典行 | 同左 |
| Qwen 四个 | 全部上文 | 否 | 没有自己的输入法 | 同左 | 同左 |
| 含章·通变 | 否 | 完整拼音 | 照青简 | 青简只拿它给整句打分 | **不跑**：它只认完整拼音 |
| 含章·知微 | 最后 64 字 | 否 | 照青简（重打分） | 青简不这样用它 | 同左 |
| 知意（青简的跑法） | 最后 64 字（默认档） | 首选的读音对齐按键 | 照青简 | 照青简 | 照青简：缩写对成 `p:` 单元 |
| 知意 int8 ONNX 两个 | 最后 128 字 | 同上（`slot_units`） | 照知意输入法 | 照知意输入法 | 照知意输入法 |

缩写打法下首选的读音不在池里，两个知意适配器从首选的字和模型自己的读音表里找出能对上按键的读音（找到的第一个；对齐只看哪些音节是全打的、缩写打了几个字母，多音字选哪个读音不影响单元）。知意输入法没有上文时不调用模型（除非装了它的 n-gram），这里照样调用。

## 复现

```sh
python eval/sentences-ctx-v1/build_set.py fetch --cache <cache>
python eval/sentences-ctx-v1/build_set.py build --cache <cache> --out <cache>/sampled.jsonl --tsv <cache>/sampled.tsv
python eval/sentences-ctx-v1/check_overlap.py --sampled <cache>/sampled.jsonl --lines c4=data/c4.txt --lines lccc=data/lccc.txt \
  --tatoeba <cmn_sentences_detailed.tsv.bz2> --wiki <zhwiki-20260901-pages-articles1.xml-p1p187712.bz2> --out eval/sentences-ctx-v1/overlap.json
# 在 MSIME 里录候选、切池（上一节）
OUT=<picks> PYTHON=<python> OURS_4M=... eval/sentences-ctx-v1/run_scorers.sh   # 变量见脚本开头
# 计时另跑一轮，在安静的机器上（pool-timing.jsonl 由 timing_pool.py 生成，已在仓库里）
OUT=<timing> TIMING_ONLY=1 PYTHON=<python> OURS_4M=... eval/sentences-ctx-v1/run_scorers.sh
python eval/sentences-ctx-v1/summarize.py --picks-dir <picks> --timing-dir <timing>
```

`build` 需要 pypinyin、jieba、opencc-python-reimplemented 和 poppler 的 `pdftotext`，都不是本仓库的依赖。政府网的页面可能改版，`sources.json` 记着取得时每个文件的哈希；对不上时，`sampled.jsonl.gz` 就是这次实际用的文字。各打分器的权重、版本和哈希与 [`eval/open-models/README.md`](../open-models/README.md) 相同，这里不重复；Qwen 用 `models.json` 里的 Hugging Face 修订号。
