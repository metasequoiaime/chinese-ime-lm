# 变更记录

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 变更

- **破坏性变更（`reference` 0.2.0）**：`ModelError` 标为 `#[non_exhaustive]`。加载器是恶意模型文件最先碰到的地方，以后在这里加校验时不应让每个匹配该枚举的调用方都跟着破坏；外部 `match` 需要带通配分支。crate 尚未发布到 crates.io，现有使用方按 git 版本锁定，不受影响。

- **破坏性变更（`reference` 0.2.0）**：`Reranker::best_where` 的第三个参数从 `impl Fn(usize) -> bool` 改为 `impl Fn(usize) -> CandidateFacts`，后者带 `answers_key` 与 `trusted_dictionary_hit` 两个字段。此前"哪些候选可比"用字数近似,"这条词典命中是否可信"用来源近似。两个近似在不做纠错时都成立、做纠错时都不成立：`xian` 读成「现」也读成「西安」,两者都覆盖全键而字数不同；而用户打错拼音时词典命中的是他没想打的那个键,词频成了另一个词的证据。两个问题只有引擎答得出,所以交给调用方而不是继续猜。`Reranker::best` 签名与行为不变,仍按那两个近似工作,文档写明限制。

- 训练进度同时写入 `progress.log` 与 `progress.json` 并逐次刷盘。此前进度只走 stdout，而长训练通常输出重定向到管道，缓冲会让进度在进程结束前完全读不到；退而求其次去读 `checkpoint.pt` 又会被误导——它只在损失改善时才写，`step` 是下界、时间戳在没有改善时停滞。

- 目录按职责重排：语料相关的三件事收进 `corpus/`（`fetch.py` 获取、`clean/` 清洗、`pinyin/` 拼音表），n-gram 的 Viterbi 解码器从 `test/` 移到 `ngram/`——它是解码器，不是测试，放在 `test/` 里与真正的 `tests/` 只差一个字母。

### 新增

- `integrations/rime/`：Rime（librime ≥ 1.17.0）插件 `ime_lm_filter`，经 C ABI 调用重排模型，只在首选是 Rime 造的句子、且至少两个候选覆盖整个分段时才可能改动首选；首选是覆盖整个分段的词典词时一律不动。librime 默认只造 1 个句子，插件要配合调高 `translator/max_sentences` 才有候选可比。附带在 Arch Linux 的 librime 上复现评测的 Docker 环境与驱动程序；在 Rime 上的实测数字尚未取得。

- `reference/ffi/`：参考实现的 C ABI（[`chinese-ime-lm-ffi` 0.1.0](https://crates.io/crates/chinese-ime-lm-ffi)，已发布到 crates.io，源码为 `96043f2` 的 `reference/ffi/`），产出静态库、动态库与 cbindgen 生成并提交的头文件 `include/chinese_ime_lm.h`。只导出 `best_where` 与 `log_probabilities`，不导出按某一个引擎的来源编号推断事实的 `best`。每个导出函数都捕获 panic、检查指针与 UTF-8，经状态码和 `cimelm_last_error` 报错；发生过 panic 的重排器拒绝后续调用，不带着可能写了一半的缓存继续运行。CI 用 `cbindgen --verify` 检查头文件与代码一致，并用真实的 C 程序链接静态库跑一遍。

- 整句重排模型权重 v1：4.25M 与 24.9M 两个规模，2026-09-18 发布于 GitHub release [`model-v1`](https://github.com/metasequoiaime/chinese-ime-lm/releases/tag/model-v1)，同一批文件也在 Hugging Face 的 [`pinyin-ime-reranker-4M`](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-4M) 与 [`pinyin-ime-reranker-25M`](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M)。它们只用于重排引擎已经产出的候选：按音节直接解码（`reference/examples/decode.rs`）的表现仍比它本想改进的引擎差，不要当解码器用。

- 参考实现以 [`chinese-ime-lm` 0.2.0](https://crates.io/crates/chinese-ime-lm) 发布到 crates.io，源码为 `795e8bf` 的 `reference/`。两张 Hugging Face 模型卡的「Use it」改为按版本号依赖，不再要求 vendor。

- `reference/README.md`：crates.io 页面展示的说明，含模型下载与校验、调用示例、门控规则和许可。crate 文档里指向仓库文件的路径改成绝对链接，在 docs.rs 上也能打开。

- `eval/probes/`：定性探针,回答"某个还不存在的功能拦在哪一层"。首个探针量的是拼音纠错要不要重训模型——不用,拦路的是候选生成与门控。它不是评测集,用例手写、候选是假设引擎会产出的样子,`eval/probes/README.md` 把这些限制写在最前面。

- 训练管线：语料构建（C4 / LCCC / 维基 / 技术文档）、字级 Transformer、训练、导出为 safetensors。
- 参考实现：纯 Rust，无 unsafe，无数值库依赖。含带门控的重排、按音节束搜索的解码器、以及混合打分的实验工具。
- 评测集：25,119 条词级用例与 60 条整句用例。
- `docs/format.md`：模型文件格式规范，供其他语言实现加载器。

### 修复

- 加载器在头部长度为 `u64::MAX` 时溢出 panic，而不是返回错误。由新增的针对性测试发现——`SECURITY.md` 声称的正是这一类不会发生。
- 张量形状相乘不做溢出检查；恶意形状溢出后得到的乘积可能恰好与真实长度相符而通过校验。配置维度现在有上界。
- 训练管线四处未用上下文管理器打开文件，会泄漏句柄。
