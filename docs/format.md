# 模型文件格式

模型是一个标准的 [safetensors](https://github.com/huggingface/safetensors) 文件，没有自定义容器。用任何语言实现加载器只需要这一篇文档。

## 文件布局

```
[0, 8)              u64（小端）  头部 JSON 的字节长度 N
[8, 8+N)            JSON        张量描述 + __metadata__
[8+N, 文件末尾)      原始数据     张量内容，偏移量相对本段起点
```

头部 JSON 的每个键是张量名，值形如：

```json
{"dtype": "F16", "shape": [12288, 448], "data_offsets": [0, 11010048]}
```

`__metadata__` 是一个字符串到字符串的字典，本模型使用以下键：

| 键 | 含义 |
|---|---|
| `format` | 恒为 `chinese-ime-lm` |
| `version` | 格式版本，当前为 `1` |
| `precision` | `f16` 或 `int8` |
| `config` | JSON 字符串，见下 |
| `vocab` | JSON 字符串数组，下标即 token id |
| `tied_embeddings` | 恒为 `true` |
| `validation_loss` | 训练时的验证损失 |
| `training_steps` | 训练步数 |
| `license` | 权重的许可 |
| `attribution` | 语料署名，**再分发时必须随权重保留** |

`config` 的字段：

```json
{"vocab": 12288, "n_layer": 8, "n_head": 8, "n_embd": 448, "context": 128, "dropout": 0.1}
```

`dropout` 只在训练时有意义，推理时忽略。

## 词表

`vocab` 是一个字符串数组，下标即 token id。前三项固定为保留 token：

| id | token | 用途 |
|---|---|---|
| 0 | `<pad>` | 填充 |
| 1 | `<unk>` | 词表外字符 |
| 2 | `<bos>` | 序列起始 |

其余每一项都是**单个**汉字。建立字符到 id 的映射时应当跳过长度不为 1 的项——保留 token 靠常量寻址，不靠文本匹配。

## 张量

| 名称 | 形状 | 说明 |
|---|---|---|
| `tok.weight` | `[vocab, n_embd]` | 字嵌入，同时用作输出投影 |
| `pos.weight` | `[context, n_embd]` | 位置嵌入 |
| `blocks.{i}.ln1.weight` / `.bias` | `[n_embd]` | 注意力前的 LayerNorm |
| `blocks.{i}.qkv.weight` | `[3*n_embd, n_embd]` | Q、K、V 合并投影 |
| `blocks.{i}.qkv.bias` | `[3*n_embd]` | |
| `blocks.{i}.proj.weight` / `.bias` | `[n_embd, n_embd]` | 注意力输出投影 |
| `blocks.{i}.ln2.weight` / `.bias` | `[n_embd]` | 前馈前的 LayerNorm |
| `blocks.{i}.fc.weight` | `[4*n_embd, n_embd]` | 前馈升维 |
| `blocks.{i}.out.weight` | `[n_embd, 4*n_embd]` | 前馈降维 |
| `ln_f.weight` / `.bias` | `[n_embd]` | 最终 LayerNorm |

权重为行主序，即 `weight[输出, 输入]`，与 PyTorch `nn.Linear` 一致：`y[o] = dot(x, weight[o]) + bias[o]`。

**`head.weight` 不在文件中。** 嵌入是绑定的，输出投影复用 `tok.weight`。

## 量化

`precision` 为 `int8` 时，二维权重按**输出行**做对称量化，并附带一个同名加 `.scale` 后缀的 float32 张量：

```
真实值[行, 列] = int8值[行, 列] * scale[行]
```

LayerNorm 参数、所有 bias、以及 `pos.weight` 保持 float32——它们占比可忽略，而对舍入最敏感。

`precision` 为 `f16` 时，除上述保持 float32 的张量外，其余为 IEEE 754 binary16。**次正规数要正确处理**：直接刷成零会引入一个调用方看不见的差异。

## 前向计算

Pre-LN 结构，因果注意力：

```
x = tok[输入] + pos[位置]

每一层：
  y = layer_norm(x, ln1, eps=1e-5)
  q, k, v = split(linear(y, qkv), 3)
  y = 因果注意力(q, k, v, 缩放 = 1/sqrt(n_embd / n_head))
  x = x + linear(y, proj)
  y = layer_norm(x, ln2, eps=1e-5)
  x = x + linear(gelu(linear(y, fc)), out)

logits = layer_norm(x, ln_f) @ tok.T
```

**GELU 必须用精确形式**（基于 erf），不是 tanh 近似——那是另一个函数，模型不是用它训练的：

```
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

## 打分

候选的分数是 `log P(候选 | 上下文)` **按字数归一**：

```
prefix = [<bos>] + 上下文字符（右截断，为候选留出位置）
分数 = (1/n) * Σ log P(候选[j] | prefix + 候选[0..j])
```

第一个字由 **prefix 最后一个位置**的隐状态预测。这一点容易漏：位置 i 预测 token i+1，所以如果只从候选自身的位置开始算，候选的首字就完全没有被打分，而后面每个字都算了。

归一是必须的，原因见 README 里的"两个陷阱"。

## 两个实现要点

**前缀可以跨按键缓存。** 已上屏文本只在上屏时改变，不是每次按键都变。缓存前缀每一层的 K/V 以及最后位置的隐状态，每次按键就只需要计算候选自身那几个字。

**点积要用多个累加器。** 累加到单个变量上会让每次乘加都等待上一次的结果，循环跑在指令延迟而非吞吐上。改成 8 个独立部分和，实测 9 候选的一次决策从 122ms 降到 26ms。
