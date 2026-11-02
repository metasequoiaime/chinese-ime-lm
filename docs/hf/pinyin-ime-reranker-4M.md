---
license: apache-2.0
language:
- zh
pipeline_tag: text-ranking
datasets:
- allenai/c4
- silver/lccc
tags:
- chinese
- pinyin
- input-method
- reranking
- character-level
- on-device
---

# pinyin-ime-reranker-4M

一个 4.25M 参数的字级 Transformer，给中文拼音输入法已经生成好的候选重新排序。磁盘上 4.5 MB，int8，一个自描述的 safetensors 文件，不带任何附属文件。

**它不做拼音解码。** 它拿到的是你的引擎已经拼好的候选列表，判断上文支持其中哪一个——也就是「引擎的第一猜测」和「用户真正想打的」之间那段差距。喂给它一串拼音，得不到中文。

训练和测量都在 [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm)。

## 两个里选哪个

| | **本模型** | [pinyin-ime-reranker-25M](https://huggingface.co/msime/pinyin-ime-reranker-25M) |
|---|---|---|
| 参数量 | 4,250,112 | 24,863,104 |
| 文件（int8） | 4.5 MB | 25.5 MB |
| 整句评测，首选（top-1）命中 | 49 / 56 | **52 / 56** |
| 每个候选字的开销 | **0.33 ms** | 2.0 ms |
| 最慢的一次按键，15 字句子，9 个候选 | **46 ms** | 268 ms |

凡是对延迟敏感的地方都选这个——移动端的键盘扩展，或者任何每次按键都要跑的路径。大模型在整句集上多对三条，在 52 条上这个差距不显著（McNemar p = 0.250，见下文），而它每次按键的开销放不进一帧。

两个文件都叫 `sentence-model.safetensors`，这是 host 要找的名字。

## 使用

参考实现用 Rust 写成，没有 `unsafe`，依赖只有 `serde`、`serde_json` 和 `thiserror`。它以 [`chinese-ime-lm`](https://crates.io/crates/chinese-ime-lm) 发布在 crates.io 上，和这些权重一样是 Apache-2.0。源码在仓库的 [`reference/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/reference) 目录，与 GPL-3.0 的训练流程分开授权，所以也可以单独拷进你的项目。

```toml
[dependencies]
chinese-ime-lm = "0.2"
```

```rust
use std::sync::Arc;
use chinese_ime_lm::{CandidateFacts, Reranker, SentenceModel};

let bytes = std::fs::read("sentence-model.safetensors")?;
let model = Arc::new(SentenceModel::load(&bytes)?);
let mut reranker = Reranker::new(model);

// Your engine's candidate list, in its own order. The returned index refers to that list;
// `None` means leave the order alone. Both facts are things only your engine can answer.
let promoted = reranker.best_where(committed_text, &candidates, |index| CandidateFacts {
    // Does this candidate answer the whole key, rather than a prefix of it or a completion
    // running past it? Not "is it the same length as the leader": `xian` reads as 现 or as
    // 西安 and both answer the key.
    answers_key: covers_whole_key[index],
    // Should the model defer to this dictionary hit? If you corrected the input, a hit on the
    // uncorrected reading carries frequency for the letters that arrived, not for the word
    // the user meant — so the answer there is no.
    trusted_dictionary_hit: sources[index] == CandidateSource::Dictionary && !corrected_key,
});
```

用其他语言实现的话，[`docs/format.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/format.md) 完整描述了文件格式和前向计算，写加载器只需要这一篇。重新实现的人几乎都会在两处栽跟头：GELU 必须用精确的 erf 形式，不能用 tanh 近似；候选的第一个字是由**前缀的最后一个位置**预测的，如果从候选自己的第一个位置开始打分，这个字就会悄无声息地漏掉不算。

## 门控是模型的一部分

重排并不是在哪里都有好处。下面两条规则由 crate 强制执行，而不是留给调用方自己记得。

**不要推翻你信任的词典命中。** 词典能答出整个输入，说明它已经带着语料里的词频，而字级模型没有这个信息。在 2,052 条这样的用例上实测：引擎自己的首选是 0.719，试过的所有阈值里重排的最好结果是 0.690。扫过二十五种「模型分 × 排名 × 词典证据」的加权方式，没有一种比「不碰词典命中」更好。

这条规则有个前提：词典回答的那个输入，就是用户想打的那个输入。**做输入纠错的引擎打破了这个前提，就得自己说出来。** 想打 `gongshi` 却打成了 `gongsi`，公司就是对实际到达的这几个字母的精确命中，带着的却是一个没人要的词的词频。所以 `trusted_dictionary_hit` 问的不是「词典有没有匹配上」，而是「模型该不该让着它」，只有调用方知道答案。

**只有覆盖整个输入的候选之间才能比较。** 引擎还会返回前缀和联想补全，它们不是首选的替代项——回答的是另一个输入。而且对数概率的总和字数越少越大，所以这里的分数是按字平均的。

**覆盖整个输入不等于长度相同，调用方得说清楚是哪一个。** 一个输入只有一种切分时，长度和这条规则一致；恰恰在纠错要紧的地方两者会分开：`xian` 可以读成「现」也可以读成「西安」，两者都用完了整个输入，按长度过滤则只会留下和首选同长的那个。所以 `best_where` 要求调用方给出 `CandidateFacts { answers_key, trusted_dictionary_hit }`，而不是自己去推断。`Reranker::best` 仍然按来源编号和字数推断这两件事，对不做纠错的引擎没问题，对做纠错的引擎就是错的。

还有一个相关的陷阱，它毁掉的是你的测量而不是准确率：按金标准长度过滤的门控在离线评测时能用，运行时用不了。用你真正要上线的判据去测，否则测出来的数字会比你的产品好看。

## 测量

60 条手写的整句用例，经由真实的输入法运行时逐键驱动，而不是离线打分。其中 56 条落在解码器那一类，只有这类才会触发重排。

| | 命中 / 56 | |
|---|---|---|
| 引擎自己的首选 | 41 | 0.732 |
| **+ 本模型** | **49** | **0.875** |
| + 25M 模型 | 52 | 0.929 |
| 金标准在候选列表内 | 52 | 0.929 |

最后一行是任何重排器的上限：剩下 4 条的正确读法从来没被拼出来过，重排器够不着。

**救回和弄坏分开报**，因为对用户来说这是两回事。本模型救回了 10 条引擎做错的，弄坏了 2 条引擎本来做对的；25M 模型救回 11 条，弄坏 0 条。所以两者之间 3 条的差距里，只有 1 条是本模型少救回的，另外 2 条是它主动弄坏的。只报一个总数，读起来是「少改进了三条」，这层区别就被藏起来了。

**在同样 52 条可达用例上，命中 48 到 52 条的模型两两之间都不显著。** McNemar 精确检验：本模型对 25M 是 0 胜 3 负，p = 0.250；对 Qwen3-0.6B 是 3 比 3，p = 1.000。它们两两之间最多只在 7 条上意见不同；只有远远落后的打分器才分得开，比如引擎原序（对 41 条；本模型 10 胜 2 负，p = 0.039）。按这个分歧率，要检出模型迭代真正关心的 1 到 2pp 差距，大约需要一千到一万条用例。60 条句子里还有十二条以 `ta` 开头、以「他」为金标准，这从拼音上根本无法判定——这些本就不该收。**不要把准确率那一列当作质量排名。** 尺寸和延迟是确定性的测量，准确率上的差别不是。

**在同一个候选池上和其他开源模型比，本模型的区别在尺寸，不在准确率。** 在这 52 条上跑了十个打分器——这里的两个模型、四个 Qwen（各按对自己最有利的起始 token 计）、青简的含章·通变和含章·知微、知意的 1.4V2-Long-L12 和 ML-pumpkin-1026——全部命中 49 到 52 条，本模型也在这个区间里。4.5 MB 的它是其中最小的，差得很远：其他输入法的模型 46 到 162 MB（知意另需一个 34 MB 的分词表），Qwen 是 278 MB 到 1.7 GB。它的延迟和其他输入法的模型在同一范围：对一个候选池做一次决策，用 `main` 上的参考实现（把每次决策的候选作为一批来算）在单个 CPU 线程上要 4.2 ms（已发布的 crate 0.2.0 是 12 ms），它们在 GPU 或四个 CPU 线程上约 5 到 10 ms。这 52 条里没有一条带上文，所以读上文的模型（包括本模型）在这里都用不上这项能力。

完整分析见 [`docs/measurements.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md#与开源通用模型比较)。

### 延迟与内存

开销与候选数、字数都接近线性，所以一个数就能概括：**每个候选字 0.33 ms，25M 是 2.0 ms**。在一台 Apple M4 Pro 上用 `reference/examples/bench.rs` 扫了两遍；32 个格子里 31 个两次结果相差在 3.5% 以内，例外的是 1 个候选 × 1 个字，怎么算都不到一毫秒。一次决策的 p50，单位毫秒：

| 候选数 \ 字数 | 1 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|
| 1 | 0.52 | 1.55 | 2.85 | 4.27 | 5.12 |
| 9 | 3.17 | 12.03 | 24.31 | 35.77 | 44.97 |

**一句话的开销不是一次决策，而是每个前缀长度各一次。** 9 个候选时，打完 15 个字共 366 ms，最慢的一次按键 46 ms——25M 则是 2,160 ms 和 268 ms。这组扫描每次按键都从头重算，因为 `bench.rs` 调用的是 `SentenceModel::score`，从不构造 `Reranker`；下面的数字是 `Reranker` 里的续算带来的变化。

在 Apple Silicon 上经由真实按键路径测量：60 条句子，1,320 次按键，配对比较。

| | p95 | 超过 16 ms 一帧的按键 | 最慢 |
|---|---|---|---|
| 每个候选从头重算 | 20.64 ms | 129 / 1320（9.8%） | 40.76 ms |
| **跨按键按位置续算** | **8.61 ms** | **1 / 1320（0.1%）** | 16.22 ms |

一次按键要打分的字里，61% 是上一次按键已经算过的某个串的前缀，所以参考实现保存每个位置的状态，只重算新增的部分。两套评测集开不开续算输出逐字节相同，不过它并不是恒等变换——浮点求和换了分组，最后几位会变。

常驻内存不等于文件大小。加载器把量化矩阵保留为 `i8`，把每行的缩放因子提到点积之外，这是同一套算术，不是近似：展开成 f32 时常驻 24.5 MB，保留 int8 时 **12.3 MB**，代价是延迟增加约 8%。在内存是硬约束的地方这笔交换很重要——iOS 键盘扩展只有几十 MB，引擎、词典和界面全在里面。

**报分位数，不报均值。** 在这个模型出身的那个输入法里，均值是 7.9 ms，而整句输入时有 25.2% 的按键超过一帧；大多数按键根本不触发重排，均值把触发那些的开销摊到了不触发的上面。

## 结构与训练

| | |
|---|---|
| 层数 / 宽度 / 头数 | 6 / 192 / 4 |
| 上下文 | 64 字 |
| 字表 | 8,192 字（覆盖率 99.987%），输入输出共享嵌入 |
| 参数量 | 4,250,112 |
| 精度 | int8，按输出行对称量化 |
| 验证集损失 | 4.274 |
| 训练步数 | 11,500 |

在同样的语料和步数预算下扫过六种形状。其中 4–7M 区间的五种彼此最多差一条，准确率分不开它们，只能由开销决定；这是五种里最便宜的。只有 24.9M 那个形状比它们高，高三条，在这套评测集上不显著。7M 到 24.9M 之间没有训练过任何形状。

在需要做决策的全部 2,105 条用例上，int8 和 float16 选的候选完全相同，所以发布的是 int8。

## 语料、许可与署名

**权重是 Apache-2.0**，和训练代码的许可不同（那个仓库是 GPL-3.0，其中 `reference/` 目录是 Apache-2.0）。大量开源输入法用的是 MIT、Apache 或 BSD 许可，链接不了 GPL 代码；一个没人能采用的公共资源，算不上公共资源。

`LICENSE` 和 `NOTICE` 放在这个仓库里，而不只是在元数据里写个名字，因为 Apache-2.0 要求再分发者把两者一并交出去，网页上的一个标签不是它们的副本。

训练语料是 [C4](https://huggingface.co/datasets/allenai/c4) 的中文部分（ODC-BY）和 [LCCC](https://huggingface.co/datasets/silver/lccc)（MIT）。**两者都没有相同方式共享（share-alike）的义务**，这正是选它们的原因：采用这个模型不会把这种义务带到你身上。语料流程也支持中文维基百科和 MDN，但特意没用，因为 CC BY-SA 会带来这种义务。

这两份语料都要求署名跟着用它们训练出来的东西走。署名写在权重文件自己的 `__metadata__` 头里，所以它和权重分不开：

```
Trained on the Chinese portion of C4 (ODC-BY, https://huggingface.co/datasets/allenai/c4)
and LCCC (MIT, https://github.com/thu-coai/CDial-GPT).
```

**再分发时请保留这个字段。** 它是从这次训练实际用到的语料推出来的，不是手写声明的：每份下载的语料旁边都会写一条记录，训练时记下语料路径，两者缺一导出就拒绝执行，不去猜。`license` 和 `attribution` 两个键在运行时可以通过 `SentenceModel::license()` 和 `SentenceModel::attribution()` 读到。

语料按行去了重。在 C4 上这删掉了一半的字，命中赌博和 SEO 关键词农场词汇的行从约 22% 降到 3.4%——重复和垃圾原来是同一个问题。

## 完整性

```
sentence-model.safetensors  4,486,280 bytes
sha256  86ac529510cb3b4968a5e6ade83ec8080f5b34a0a681e75c362accbbd387d1c1
```

模型文件是输入法加载进自己进程的一个不可信二进制。发布之前先校验摘要。参考加载器把头部里的每个维度都当作恶意输入，在做任何乘法展开之前先限定范围；见 [`SECURITY.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/SECURITY.md)。

## 评测集

评测集也许比权重更有用：25,119 条词级用例和 60 条整句用例，在 [`eval/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval)。上面说的两个陷阱都已经固化成其中的固定用例，而不是留在文档里当建议。
