# 安全向量量化点积的机器码调查

固定本库 `main` 的 `4e745be72f1c49e9c7b5dadea819fbb1fcda0e13`，主项目 `6d9cb3da7fa6cd85baf7abce937b4fa7a252cb5a`、模型 pin `15caab4fdeb59c11f452b91884371cf30e2e5608`。本库基线相对该 pin 的生产源码没有差异，只有 workflow 和依赖更新。来源为实际远端默认分支与固定提交，没有复制相邻仓库的未提交内容。

## 原型

`wide` 0.7.33 临时作为 dev-dependency，原型仅在诊断测试模块中，用 `f32x8`、`i32x8::from(&[i8])` 和 `round_float` 表达原来的八路累加。保留归约与尾部顺序、int8 常驻表示，没有 FMA、unsafe 或架构分支。生产点积从未替换成原型。

冻结旧函数和原型各通过一项聚合逐位对照：十种基础长度、每侧四种附加长度，以及宽指数/非整数尾数和负零、正负无穷、NaN。完整源码随目录交付；这不等于模型评分/分布的完整证明，也不覆盖所有特殊 NaN 载荷。原型测试的 unused import 编译警告保留，没有声称原型 Clippy 通过。

[官方版本元数据](https://crates.io/api/v1/crates/wide)中，0.7.33 未撤销并声明 Rust 1.61；0.7.34 已撤销且要求 1.89，0.8 系列要求 1.89。读取到的相关版本信息保存在 `wide-versions.json.gz`。旧兼容系列的 `From<&[i8]>` 以及 ARM64 的 `from_u8x16_low`、`from_i16x8` 都经普通数组转换；只核对这条其它 API 路径，没有实施第二原型。未以当前编译器的成功代替实际 Rust 1.75 验证。

## 机器码

本机 ARM64，Rust 1.97.1、LLVM 22.1.6，Cargo 默认 release，无额外 target-cpu 或 fast-math。两份程序的二进制与探针 SHA-256、恢复源码哈希见 `binaries.json.gz` 和 `provenance.json.gz`，没有分发程序。

| 每八个元素的循环 | 冻结旧入口对应的生产点积 | 显式向量原型 |
| --- | ---: | ---: |
| 全部指令 | 12 | 14 |
| `tbl.16b` | 2 | 2 |
| `scvtf.4s` | 2 | 2 |
| `fmul.4s` | 2 | 2 |
| `fadd.4s` | 2 | 2 |
| 独立指针 `add` | 0 | 2 |

原型没有消除目标转换/打包操作，加载后的指针更新表达还有差别。完整反汇编保留在 `red-dot.s.gz` 与 `candidate-dot.s.gz`，循环边界和计数在 `loop-analysis.json.gz`。指令数不是时间，测试包装器也不是主项目发布程序，不能给出产品快慢的百分比。

## 没有产生性能成绩

旧入口预算等待连续 30 秒无编译且一分钟 load<=10，180 秒未满足；`red-run.json.gz` 保存全部等待记录，`timed=false`。没有运行旧入口计时，不存在性能红测失败，也没有候选计时。没有删去失败或超时记录。

主要假设没有获得支持，本轮不新增生产依赖或点积实现，没有继续等候或运行完整 linear/project 配对、模型 bench 整表、MSRV、跨平台编译、产品质量、九键 ABBA 或主项目 quick。没有宣称所有安全 SIMD 库都无效，也没有声称九键 16 ms 已达标。

## 复现

在包含本目录的仓库外独立检出、macOS ARM64 和同一 Rust 工具链上运行。脚本验证源码、清单和锁文件等于固定基线，以已冻结补丁临时接线、构建两份独立程序并运行逐位检查和反汇编，`finally` 逐字恢复三份文件；只使用默认 `reference/target/`。

```sh
python3 docs/measurements/2026-10-23-safe-vector-dot/reproduce.py --check-only
python3 docs/measurements/2026-10-23-safe-vector-dot/reproduce.py
python3 docs/measurements/2026-10-23-safe-vector-dot/analyze.py
```

复现脚本用 Cargo JSON 读取可执行文件路径，避免依赖本机产物名字。它的构建只指定 lib test，与实际调查的默认 no-run 相比省去未用于调查的集成测试产物；原始实际执行脚本另存于 `build_candidate.py.gz`，该文件保留历史二进制名字。复现不保证另一编译环境的二进制字节相同。交付脚本已实际重放成功，两项逐位检查通过，重放机器码的循环指令及操作计数与原记录相同；重放记录随目录交付。`analyze.py --recorded` 可直接重算交付的原反汇编。

若要重访局部内核配对，先恢复原定门槛、再使用已交付的等待程序，不得把这些内核结果冒充完整模型结果：

```sh
gzip -dc docs/measurements/2026-10-23-safe-vector-dot/run.py.gz > reference/target/safe-dot-evidence/run.py
python3 reference/target/safe-dot-evidence/run.py red reference/target/safe-dot-evidence/red-binary
python3 reference/target/safe-dot-evidence/run.py candidate reference/target/safe-dot-evidence/candidate-binary
```

恢复后的 `cargo test --locked`、全目标 Clippy `-D warnings`、Rustfmt 和差异检查通过；单测 45 通过、12 忽略，模型文件测试 15 通过，doc tests 0 项。没有提供真实模型路径，不声称执行了忽略的真实模型回归。本库没有主项目的 quick 与 pnpm 笔记入口，笔记通过主项目现有的三份独立校验器在本库根目录校验。

`manifest.json` 保存原始、脱敏后及 gzip 字节的 SHA-256，压缩使用固定 `mtime=0`。本机路径替换为 `<checkout>`、`<user-home>`；内部原始哈希与 manifest 的 `raw_sha256` 关联，不直接和脱敏文件比较。交付只有合成诊断、文档和日志，没有模型、词表、真实输入、凭据或账号资料。
