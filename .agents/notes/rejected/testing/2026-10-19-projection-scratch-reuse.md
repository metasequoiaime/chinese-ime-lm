# Agent Note: 词表投影 logits 缓冲复用

Status: rejected — 局部配对测量没有稳定收益，恢复生产实现

## Problem

`SentenceModel::log_probability` 只需要 `project` 的归一化常数，但 `project` 同时分配并填充整张词表 logits。候选一次前向会为多个隐藏状态执行这条路径，因此调查是否能在一次 `run_tail` 内复用 logits 缓冲，减少临时分配。

## Proposal

新增 `ProjectionScratch`，把 `project` 的写入拆为 `project_into`；`run_tail` 为整批隐藏状态持有一个可复用的 `Vec<f32>`，保持 Float/Quantized 分支、逐行点积、最高值和指数求和顺序不变。公开分布仍返回独立 logits，归一化位模式保持逐位一致。

## Verification

先加入行为测试：无 logits 辅助路径与完整投影的归一化逐位相同；连续投影复用同一缓冲地址。测试在旧实现上按预期无法编译，加入最小实现后通过。原型在 Rust release 下运行参考仓库完整测试：46 个普通单测、15 个模型文件测试和 doc tests 全部通过，10 个既有性能测试保持忽略。

局部配对计时使用合成 Float/Quantized 权重，宽度 192/768、词表 8192、8 个不同隐藏状态、每臂 50×8 次、交替先后。中位新旧比值为：Quantized/192 `0.9994`、Float/192 `0.9973`、Quantized/768 `1.0021`；其余形状在本次独占计时窗口内未完成。收益低于噪声，且宽 768 量化形状回退，不能建立安全门槛或端到端收益。没有把这次局部结果当成 16ms 证据。

## Decision

恢复 `reference/src/lib.rs` 与 `reference/src/tests/project_rows.rs` 到 `origin/main`，不提交生产改动，也不更新主项目模型 pin。后续若要继续，应先找到能减少完整投影计算量的方案；单纯复用这一小段 logits 分配不足以成为独立性能片。
