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

# pinyin-ime-reranker-25M

一个 24.9M 参数的字级 Transformer，给中文拼音输入法已经生成好的候选重新排序。磁盘上 25.5 MB，int8，一个自描述的 safetensors 文件，不带任何附属文件。

**它不做拼音解码。** 它拿到的是你的引擎已经拼好的候选列表，判断上文支持其中哪一个——也就是「引擎的第一猜测」和「用户真正想打的」之间那段差距。喂给它一串拼音，得不到中文。

训练和测量都在 [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm)。

## 选它之前先看这一节

这是两个里大的那个。在整句评测集上，它拿下了重排器够得着的每一条，比 4.25M 的兄弟模型多对三条——这个差距在这套评测集上不显著（McNemar p = 0.250，见下文）。它每次决策的开销也是 4.25M 的**六倍**，两个都是直接实测的。

| | **本模型** | [pinyin-ime-reranker-4M](https://huggingface.co/msime/pinyin-ime-reranker-4M) |
|---|---|---|
| 参数量 | 24,863,104 | 4,250,112 |
| 文件（int8） | 25.5 MB | **4.5 MB** |
| 整句评测，首选（top-1）命中 | **52 / 56** | 49 / 56 |
| 每个候选字的开销 | 2.0 ms | **0.33 ms** |
| 最慢的一次按键，15 字句子，9 个候选 | 268 ms | **46 ms** |

**如果模型要在每次按键时跑，选小的那个。** 移动端键盘扩展负担不起这个文件，也负担不起这个延迟。这个模型用在桌面上、不是每键都跑的路径上，或者当作衡量小模型的准确率参照。

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
| + 4.25M 模型 | 49 | 0.875 |
| **+ 本模型** | **52** | **0.929** |
| 金标准在候选列表内 | 52 | 0.929 |

最后两行相等，就是全部的结论。52 是任何重排器的上限——另外 4 条的正确读法从来没被拼出来过，所以任何只给列表重新排序的东西都够不着（金标准「李明在北京工作」，候选全是「利民」）。本模型拿下了每一条可达用例，这意味着**这套评测集已经没有余地显示出更好的重排器**：在它上面，下一个能测出来的改进只能来自候选生成。

它救回了 11 条引擎做错的，弄坏了 0 条引擎做对的。4.25M 模型救回 10 条、弄坏 2 条，所以 3 条的差距里只有 1 条是它少救回的，2 条是它弄坏的。把两者分开很重要，因为只报一个总数读起来是「救回的少一些」，而用户感受到的是「本来对的变错了」。

### 52 / 52 不代表质量

有两件事限制了这个数字，而且指向同一个方向。

**命中 48 到 52 条的模型之间，没有一对差异显著。** McNemar 精确检验，52 条可达用例：本模型对 4.25M 是 3 胜 0 负，p = 0.250——三次同方向的判定，和抛三次硬币都是正面的概率一样。对 Qwen3-0.6B、Qwen3.5-0.8B 和 Qwen3.5-2B 分别是 3–0、3–0 和 2–0，p = 0.250、0.250 和 0.500。它们两两之间在 52 条里最多只有 7 条意见不同。只有比本模型落后 10 条或更多的打分器才和它分得开：引擎原序（11–0，p = 0.001），以及最初测量时的 Qwen2.5-0.5B（10–0，p = 0.002）。按这个分歧率，要检出模型迭代关心的 1 到 2pp 差距，大约需要一千到一万条用例。另外两个开源输入法的重排模型也落在同一区间：青简的含章·通变 50、含章·知微 49，知意的 1.4V2-Long-L12 51、ML-pumpkin-1026 49，与本模型都没有显著差异。它们是 46 到 162 MB，本模型 25.5 MB；每次决策它们更快：在 GPU 或四个 CPU 线程上约 5 到 10 ms，本模型用 `main` 上的参考实现（把每次决策的候选作为一批来算）在单个 CPU 线程上要 23 ms（已发布的 crate 0.2.0 是 72 ms）。这 52 条里没有一条带上文。

**分数里有一部分是语料先验，不是排序能力。** 60 条句子里有十二条以 `ta` 开头、以「他」为金标准。他、她、它在拼音上无法区分，没有上文时从信息论上讲就不存在正确答案——模型只能顺着训练语料的先验走。本模型答对它们，是因为在 C4 和 LCCC 里「他」排在「她」前面，恰好和写用例的人一致。按项目自己的收录规则，这 12 条本就不该收，去掉之后只剩 40 条真正可判定的用例。

所以：把 52 / 52 理解成「本模型在一个规模小、部分不可判定的集合上没有弄坏任何东西」，而不是一个需要捍卫的准确率数字。**尺寸和延迟是确定性的测量，没有置信区间；准确率那一列不是。** 完整分析见 [`docs/measurements.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/docs/measurements.md)，其中包括在同一个候选池上与四个开源 Qwen 模型、另外两个开源输入法的重排模型的比较。

### 延迟与内存

开销与候选数、字数都接近线性，所以一个数就能概括：**每个候选字 2.0 ms，4.25M 是 0.33 ms**——六倍。两个模型都在一台 Apple M4 Pro 上用 `reference/examples/bench.rs` 扫了两遍，只引用两次结果一致的格子；本模型的 32 个格子全部相差在 3.0% 以内。

一次决策的 p50，单位毫秒：

| 候选数 \ 字数 | 1 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|
| 1 | 3.52 | 9.43 | 17.33 | 25.16 | 31.10 |
| 9 | 19.57 | 73.71 | 145.41 | 216.18 | 270.15 |

**一句话的开销不是一次决策，而是每个前缀长度各一次**：打 n 个字，要为长度 1 到 n 各付一次。9 个候选时，打完 15 个字本模型共 2,160 ms，4.25M 是 366 ms；最慢的一次按键 268 ms，对 46 ms。

**这就是本模型上不了按键路径的原因**，用不着看分位数：9 个候选、六个字的一次决策就要 109 ms，而一帧是 16 ms。

**早先那个真实运行时的数字来自另一套测量装置，不能和上表比较。** 经由一个真实输入法驱动、每次按键把每个候选从头重算时，本模型的 p95 是 97.2 ms，40.4% 的按键超过一帧；4.25M 是 20.3 ms 和 10.1%。参考实现后来加上了跨按键按位置续算，在同样 1,320 次按键上把 4.25M 的 p95 从 20.64 ms 降到 8.61 ms。本模型从来没有这样重测过，因为那需要那个运行时录下的候选列表，而不是一组合成的扫描。可以预期改进的形状相近，但不要假定一个数字。

`bench.rs` 调用的是 `SentenceModel::score`，从不构造 `Reranker`，所以上表按定义就是不续算的开销——和 97.2 ms、20.3 ms 那两个数是同一种口径，而不是 8.61 ms 那种。

常驻内存不等于文件大小。加载器把量化矩阵保留为 `i8`，把每行的缩放因子提到点积之外，这是同一套算术，不是近似。早先展开成 f32 的加载器把这个 25.5 MB 的文件变成了 **126 MB 常驻**；在当时测量的 4.25M 模型上，保留 int8 大约把常驻减半（24.5 MB 到 12.3 MB，代价是延迟增加约 8%），但本模型对应的数字没有记录。

**报分位数，不报均值。** 大多数按键根本不触发重排，均值把触发那些的开销摊到了不触发的上面。

## 结构与训练

| | |
|---|---|
| 层数 / 宽度 / 头数 | 8 / 448 / 8 |
| 上下文 | 128 字 |
| 字表 | 12,288 字（覆盖率 99.999%），输入输出共享嵌入 |
| 参数量 | 24,863,104 |
| 精度 | int8，按输出行对称量化 |
| 验证集损失 | 3.851 |
| 训练步数 | 11,000 |

这是 `neural/model.py` 里的 `desktop` 预设。在同样的语料和步数预算下扫过六种形状；只有这一种比 4–7M 那一档高，高出的三条在这套评测集上不显著，代价是延迟——它与已发布的 4.25M 之间实测的倍数见上一节。7M 到 24.9M 之间没有训练过任何形状。

**容量是手头最有效的杠杆。** 同样的语料下，这个形状在 5,000 步时的困惑度，已经低于 6×256 键盘形状在 50,000 步时达到的水平。

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
sentence-model.safetensors  25,480,184 bytes
sha256  0a6ecba69bf1d39c7eb49549c716477dd6c03fdf05757773e1f50435262fb469
```

模型文件是输入法加载进自己进程的一个不可信二进制。发布之前先校验摘要。参考加载器把头部里的每个维度都当作恶意输入，在做任何乘法展开之前先限定范围；见 [`SECURITY.md`](https://github.com/metasequoiaime/chinese-ime-lm/blob/main/SECURITY.md)。

## 评测集

评测集也许比权重更有用：25,119 条词级用例和 60 条整句用例，在 [`eval/`](https://github.com/metasequoiaime/chinese-ime-lm/tree/main/eval)。上面说的两个陷阱都已经固化成其中的固定用例，而不是留在文档里当建议。
