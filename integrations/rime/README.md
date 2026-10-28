# Rime 插件：ime_lm_filter

给 [Rime](https://rime.im)（librime）用的候选重排插件。它在 Rime 已经排好的候选上调用 chinese-ime-lm 的整句重排模型，按 [`docs/measurements.md`](../../docs/measurements.md) 测出来的门控规则决定要不要把另一个候选提到首位。它通过 [`reference/ffi`](../../reference/ffi) 的 C ABI 调用模型。

**它不解码拼音，也不改变候选集合。** 它只在 Rime 给出的候选里挑首选，其余候选保持 Rime 原来的顺序。

## 它什么时候会动首选

只有同时满足下面几条时才会改动首选，任何一条不满足都保持 Rime 的顺序：

- 首选是 Rime 造的句子（`sentence`），而不是词典词。librime 的拼音翻译器只在没有可靠的精确词典命中时才造句；首选是覆盖整个分段的词典词（`phrase`、`user_phrase`、`table`、`user_table`）时，插件一律不动。这对应测量里「不要推翻可信的词典命中」那条规则。首选是纠错得来的词典词时也不动，宁可错过一次纠正，也不冒改错的风险。
- 至少有两个候选覆盖整个分段。前缀和联想补全回答的是另一个键，不参与比较。
- 模型认为另一个覆盖整个分段的候选比首选更符合前文。

第二条意味着：**librime 默认每次只造 1 个句子（`translator/max_sentences: 1`），这时首选几乎没有可比的对手，插件基本不会起作用。** 要让它有候选可选，需要把 `max_sentences` 调高，例如 5。这个选项从 librime 1.17.0 开始提供。

## 要求

- librime 1.17.0 或更新。插件用的是 librime 的 C++ 内部接口，必须对照将要加载它的那个 librime 版本编译。
- 编译时需要 Rust 工具链（1.75 或更新），由 CMake 调用 cargo 编译 `reference/ffi`。没有 Rust 工具链时，可以把 `CIMELM_FFI_LIBRARY` 和 `CIMELM_FFI_INCLUDE_DIR` 指向预先编好的 `chinese-ime-lm-ffi` 静态库和头文件。
- 模型文件 `sentence-model.safetensors`。推荐用 4M 模型（[GitHub release `model-v1`](https://github.com/metasequoiaime/chinese-ime-lm/releases/tag/model-v1) 或 [Hugging Face](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-4M)），sha256 为 `86ac529510cb3b4968a5e6ade83ec8080f5b34a0a681e75c362accbbd387d1c1`。

## 编译

librime 的插件要放在它源码树的 `plugins/` 目录下，和 librime 一起编译。本目录不是独立仓库，所以不能用 `RIME_PLUGINS=<仓库>` 自动拉取，要手动链接进去：

```sh
git clone --branch 1.17.0 https://github.com/rime/librime.git
ln -s /path/to/chinese-ime-lm/integrations/rime librime/plugins/ime-lm
```

然后按 librime 自己的方式编译：

- **合并进 librime**（librime 默认，Squirrel 和 Weasel 的官方包也是这样）：照常编译 librime，插件会编进 librime 本体。
- **外部插件**（Arch Linux 的 librime 就是这样打包的）：用 `-DBUILD_MERGED_PLUGINS=Off -DENABLE_EXTERNAL_PLUGINS=On` 配置，编译目标 `rime-ime-lm`，得到 `lib/rime-plugins/librime-ime-lm.so`，放进系统 librime 的插件目录（Arch 上是 `/usr/lib/rime-plugins/`）。librime 在 Windows 上还不支持外部插件，Windows 只能用合并方式。

[`eval/run.sh`](eval/run.sh) 就是在 Arch Linux 上按外部插件方式完整走一遍的例子。

## 启用

在要用的输入方案的补丁文件里（例如 `luna_pinyin_simp.custom.yaml`）加入：

```yaml
patch:
  translator/max_sentences: 5
  engine/filters/+:
    - ime_lm_filter
  ime_lm_filter/model: sentence-model.safetensors
```

- `ime_lm_filter` 追加在已有的 filter 之后，这样模型看到的是简繁转换之后的候选文字。模型是在简体语料上训练的。
- `ime_lm_filter/model` 是相对路径时，先在用户目录里找，找不到再到共享数据目录里找；也可以写绝对路径。默认值是 `sentence-model.safetensors`。
- 模型加载失败时，插件会在 librime 的日志里报错，然后把候选原样放行，不影响输入。

上下文取自当前编码里前面分段已选的文字，或者本次会话最近提交的文字。

## 评测

[`eval/`](eval/) 在 Arch Linux 自带的 librime 上复现评测：编译插件、装进系统插件目录，然后用 [`eval/sentences-v1.tsv`](../../eval/sentences-v1.tsv) 的 60 句整句用例，通过 librime 的 C API 跑朙月拼音简化字（`luna_pinyin_simp`）的四种配置，逐个用例对比首选：

```sh
bash integrations/rime/eval/run.sh
```

需要 Docker。Arch Linux 只发布 amd64 镜像，在 arm64 机器上会走模拟，编译 librime 要花很久。

目前还没有在 Rime 上测得的数字。这个模型的收益是在水杉输入法的引擎上测出来的，在 Rime 上是否成立，要以 `eval/run.sh` 的结果为准。

## 许可

Apache-2.0，和 `reference/` 相同，见 [`LICENSE`](LICENSE)。
