# 与其他模型在同一个候选池上比较

[`docs/measurements.md`](../../docs/measurements.md#与开源通用模型比较) 里「与开源通用模型比较」那张表的全部输入、打分器和结果。每个打分器拿到同一个池，每条用例从池里挑一个候选，挑中金标准算对。

**这里只量「在同一个池里挑一个」这一项能力。** 别的输入法的模型是给它们自己的候选生成配套训练和调参的，在它们自己的输入法里还和别的打分项一起用；这里的数字不代表它们在自己输入法里的表现。

## 文件

| 文件 | 内容 |
|---|---|
| `pool.jsonl` | 52 条用例的候选池，引擎原序。每行 `id`、`input`、`gold`、`syllables`、`tags`、`context`（全为空）、`candidates`、`sources` |
| `build_pool.py` | 从 MSIME `convert_eval --dump` 切出池 |
| `../../reference/examples/pool.rs` | 本项目的模型（参考实现） |
| `mlx_lm_pool.py` | 通用语言模型（Qwen），经 `mlx-lm` |
| `qingjian_pool.rs` | 青简的三个模型，用青简自己的推理代码；要放进青简的源码树编译 |
| `zhiyi_onnx_pool.py` | 知意的 int8 ONNX，按知意输入法的序列与会话设置 |
| `models.json` | 每个打分器的来源、版本、哈希、体积、运行方式，以及它的逐例结果在哪个文件 |
| `summarize.py` | 把逐例结果汇成 `results.json` 与 `results.csv` |
| `results.json` | 每个打分器的命中、Wilson 95% 区间、每轮的决策耗时中位数、逐例选择、错例，以及对引擎原序和本项目两个模型的 McNemar 精确检验 |
| `results.csv` | 同上的一行一个打分器，不含逐例选择 |

别的项目的权重和代码都不在这里：要复现就按下面的来源自己下载，核对哈希。`qingjian_pool.rs` 是本项目写的文件（Apache-2.0），只在青简的源码树里和青简的 crate 一起编译。

## 池

`pool.jsonl` 由 2026-09-19 当天的 MSIME（`dcdb815df26da8968012710ac7097c41bec9987f`）录出，就是原表那次测量用的引擎和词库（`desktop-dictionary.lock.json`：dict-v1.0.0）。资源目录里不放 `sentence-model.safetensors`，所以录下来的是引擎自己的顺序：

```sh
# MSIME dcdb815 的源码树里
cargo build --release -p msime-input-runtime --example convert_eval
target/release/examples/convert_eval --resources <按锁取齐、去掉模型的资源目录> \
  --set resources/eval/sentences-v1.tsv --dump engine-dump.jsonl
python eval/open-models/build_pool.py --dump engine-dump.jsonl --set eval/sentences-v1.tsv --out eval/open-models/pool.jsonl
```

MSIME 那一版的 `engine-lock.json` 锁着 MSIME-Engine `0531d42` 源码包的 SHA-256 `819ea317…`，GitHub 现在给出的同一个提交的源码包是 `352dbebc…`，要先改掉锁里的这一项才能编译；其余依赖的哈希不变。`sentences-v1.tsv` 与本仓库 `eval/` 下的那份逐字节相同。

切池的规则：取 `generated`（8）和 `fallback`（9）两个来源、字数与金标准相同的候选，保持引擎顺序；金标准不在池里的 8 条丢掉。剩 52 条，每条 6 到 7 个候选，平均 6.27。引擎原序 41 条对，本项目 4.25M 49 条、24.9M 52 条，与原表逐条相同。

## 打分器

每个打分器每条用例输出一行 JSON（`id`、`pick`、`ms`），按池的顺序。下面的命令把它们写进同一个目录，`summarize.py` 从那里读，文件名就是 `models.json` 里的 `picks` 与 `timing`。

### 本项目的模型

```sh
cd reference
cargo run --release --example pool -- sentence-model.safetensors ../eval/open-models/pool.jsonl --passes 5 > out/ours-4m.r1.jsonl
cargo run --release --example pool -- sentence-model-desktop.safetensors ../eval/open-models/pool.jsonl --passes 5 > out/ours-25m.r1.jsonl
```

挑每字平均对数概率最高的候选（并列取靠前的），与 `Reranker::best_where` 决定比较之后的做法相同。耗时是对整个池调用一次 `SentenceModel::score`，单线程、不续算。

### Qwen

```sh
python eval/open-models/mlx_lm_pool.py --model mlx-community/Qwen3-0.6B-8bit --pool eval/open-models/pool.jsonl --start none > out/qwen3-0.6b.none.jsonl
```

`--start` 决定候选前面接什么：`none` 什么也不接，第一个 token 不计分，这是 2026-09-19 原表的口径；`endoftext` 接 `<|endoftext|>`；`eos` 接分词器的 `eos_token`（这几个版本是 `<|im_end|>`）。结果对这一项很敏感，`docs/measurements.md` 里有数字。原来的打分脚本没有留下，这一个是重写的，按 `none` 复测时 0.5B、0.6B、2B 的命中数与原表相同，0.8B 少一条。模型的 Hugging Face 修订号记在 `models.json` 里。

### 青简：含章·通变、含章·知微、知意 safetensors

用青简自己的推理代码跑，不重写。把 `qingjian_pool.rs` 放进青简 `b7a4f8bbb124c7e4df7660f6c2ce97b627113205` 的源码树，按青简 macOS 版的特性编译：

```sh
cp eval/open-models/qingjian_pool.rs <qingjian>/crates/qingjian-neural/examples/ime_lm_pool.rs
cd <qingjian>
cargo build --release -p qingjian-neural --example ime_lm_pool --features metal,zhiyi
target/release/examples/ime_lm_pool tongbian <hanzhang-tongbian-small.qjm> pool.jsonl > out/hanzhang-tongbian.r1.jsonl
target/release/examples/ime_lm_pool zhiwei <hanzhang-zhiwei-small.qjm> pool.jsonl > out/hanzhang-zhiwei.r1.jsonl
target/release/examples/ime_lm_pool zhiyi <zhiyi 目录> pool.jsonl > out/zhiyi-candle.r1.jsonl
```

各个模型拿到的就是青简引擎给它的：

- **含章·通变**：`P2cScorer` 的 `SentenceScorer::score`，即 `log P(候选 | 敲的字母)`，不读前文。候选里有字表外的字时它不打分、引擎保持原序；适配器照做（选池里第一个、记为 declined），这个池里没有发生。
- **含章·知微**：`CharScorer` 的 `SentenceScorer::score`，即 `log P(候选 | 前文)`，不看字母。
- **知意**：`ChoiceScorer` 经 `ChoiceModel`，与 `Engine::rerank_choices` 的调用相同：输入单元是首选候选的音节与敲的字母对齐（`ChoiceModel::align`），一次调用给整个池按池的顺序打分，取 logit 最高的。首选的音节不在池里，适配器把字母切成该用例音节数个音节（用模型自己的单元表）；切法不止一种时，取首选候选的字能读成的那一种（用模型自己的 `readings`）。池里有三条用例有两种切法（`s-010`、`s-018`、`s-024`），换另一种切法或干脆不给单元，这三条的选择都不变。

不带 `metal` 编译就是同一份代码在 CPU 上跑；选择逐例相同。

### 知意 int8 ONNX

```sh
python eval/open-models/zhiyi_onnx_pool.py --model-dir <含 laya.int8g.onnx 的目录> --check test_vectors.safetensors
python eval/open-models/zhiyi_onnx_pool.py --model-dir <含 laya.int8g.onnx 的目录> --pool eval/open-models/pool.jsonl > out/zhiyi-onnx.r1.jsonl
```

序列照知意输入法 `laya::Reranker::build` 的 `ctx-first` 分支拼，输入单元照 `slot_units`，会话设置照它的 CPU 会话（4 个 intra-op 线程、1 个 inter-op 线程、全部图优化），ONNX Runtime 用它锁定的 1.24.4。`--check` 读上游 safetensors 包（知意 Release `model-zhen-long-l12` 的 `laya-zhen-long-l12-safetensors.zip`）里的 `test_vectors.safetensors`：四条中文选词用例从明文拼出的 token、单元和候选标记位置与测试向量逐个相同，int8 图与 fp32 图的首选一致（logit 最大差 0.16）。

知意输入法在没有前文时不调用模型，除非装了它的词级 n-gram；它还把模型分与 n-gram、位置先验和组句分相加。这些都不是模型本身，这里不用。

### 汇总

```sh
python eval/open-models/summarize.py --picks-dir out
```

## 来源

| 模型 | 来源 | 文件 SHA-256 | 许可 |
|---|---|---|---|
| 含章·通变 | [qingjian-models `hanzhang-tongbian-v1.0.0`](https://github.com/qingjian-team/qingjian-models/releases/tag/hanzhang-tongbian-v1.0.0)，分块 `8661aea7…` | `hanzhang-tongbian-small.qjm` `4eabcb14…` | GPL-3.0-or-later |
| 含章·知微 | [qingjian-models `hanzhang-zhiwei-v1.0.0`](https://github.com/qingjian-team/qingjian-models/releases/tag/hanzhang-zhiwei-v1.0.0)，分块 `3fa74002…` | `hanzhang-zhiwei-small.qjm` `83ad55ab…` | GPL-3.0-or-later |
| 知意 1.4V2-Long-L12 | [qingjian-models `zhiyi-v1.0.0`](https://github.com/qingjian-team/qingjian-models/releases/tag/zhiyi-v1.0.0)，整包 `4fca2aad…`；与[知意 `model-zhen-long-l12`](https://github.com/dd1000001000/zhiyi_ime/releases/tag/model-zhen-long-l12) 的 safetensors 逐字节相同 | `model.safetensors` `c975b57b…` | Apache-2.0 |
| 知意 1.4V2-Long-L12 int8 | [qingjian-models `zhiyi-onnx-v1.0.0`](https://github.com/qingjian-team/qingjian-models/releases/tag/zhiyi-onnx-v1.0.0)，分块 `d693c2a2…`；即知意输入法 1.5.18 锁定的图 | `laya.int8g.onnx` `82d0fd05…` | Apache-2.0 |
| 知意 ML-pumpkin-1026 int8 | [知意 `model-ml-pumpkin-1026`](https://github.com/dd1000001000/zhiyi_ime/releases/tag/model-ml-pumpkin-1026)，`laya-ml-pumpkin-1026-int8g.zip` `f73c380b…`；知意输入法 1.6.0 起的默认模型 | `laya.int8g.onnx` `bbf51918…` | Apache-2.0 |

完整哈希在 `models.json`。读过的代码：青简 `b7a4f8b`（`crates/qingjian-neural`、`crates/qingjian-core/src/engine/choice`、`engine/rescoring`、`engine/query/generating.rs`），qingjian-models `3fb2831`，知意输入法 `935226b`（`engine/src/laya/`、`engine/src/laya_rerank.cc`、`scripts/fetch_model.py`）。

## 找过、没有量的

- **言泉输入法（Cassotis IME，GPL-3.0）**：约 1.5 亿参数的字级语言模型给整句候选打分，再由梯度提升的策略模型结合引擎特征决定首选。它的 ONNX 图用了自定义算子 `cassotis:TrieAttention`，只在它的 Windows 桥接库里实现，在这台机器上跑不了原图；决定首选的又不是模型本身。
- **Beam 输入法（Apache-2.0）**：Qwen3-0.6B-Base 微调的生成模型，按敲的键和前文直接写出整句，不给外来的候选打分；在池里挑一个不是它的用法。
- **PinyinGPT**（Transformers4IME，ACL 2022，CC BY-NC-SA 4.0）：论文的研究模型，没有找到用它的输入法，也就没有一个「它自己的输入法怎么喂它」可以照做。
- **拉風輸入法、樹懶智慧輸入法、Qwen3.5-TW-IME**：输入是注音、输出是繁体，与这套全拼简体的用例不对应。
