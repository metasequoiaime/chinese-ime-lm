# 变更记录

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增

- 训练管线：语料构建（C4 / LCCC / 维基 / 技术文档）、字级 Transformer、训练、导出为 safetensors。
- 参考实现：纯 Rust，无 unsafe，无数值库依赖。含带门控的重排、按音节束搜索的解码器、以及混合打分的实验工具。
- 评测集：25,119 条词级用例与 60 条整句用例。
- `docs/format.md`：模型文件格式规范，供其他语言实现加载器。

### 尚未提供

- **发布的模型权重。** 训练仍在进行。在当前模型上，按音节直接解码的表现比它本想改进的引擎还差，发布这样的权重对一个公共资源是负资产。
- crates.io 发布。
