use super::*;
use std::hint::black_box;
use std::time::Instant;

// 固定 cc1d3dd 的完整前馈、线性层、归一化和前向路径，计时包含原来的申请与析构。
fn reference_feed_forward(model: &SentenceModel, x: &mut [f32], block: &Block) {
    let n_embd = model.config.n_embd;
    let normed = reference_layer_norm(x, &block.ln2_weight, &block.ln2_bias, n_embd);
    let mut hidden = reference_linear(
        &normed,
        &block.fc_weight,
        &block.fc_bias,
        n_embd,
        4 * n_embd,
    );
    for value in &mut hidden {
        *value = gelu(*value);
    }
    let out = reference_linear(
        &hidden,
        &block.out_weight,
        &block.out_bias,
        4 * n_embd,
        n_embd,
    );
    for (slot, value) in x.iter_mut().zip(out) {
        *slot += value;
    }
}

fn reference_linear(
    x: &[f32],
    w: &Matrix,
    bias: &[f32],
    inputs: usize,
    outputs: usize,
) -> Vec<f32> {
    let rows = x.len() / inputs;
    let mut out = vec![0.0; rows * outputs];
    for row in 0..rows {
        let source = &x[row * inputs..(row + 1) * inputs];
        let target = &mut out[row * outputs..(row + 1) * outputs];
        for (index, slot) in target.iter_mut().enumerate() {
            *slot = w.dot_row(source, index, inputs) + bias[index];
        }
    }
    out
}

fn reference_layer_norm(x: &[f32], weight: &[f32], bias: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (source, target) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        let mean = source.iter().sum::<f32>() / width as f32;
        let variance = source
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + 1e-5).sqrt();
        for (index, slot) in target.iter_mut().enumerate() {
            *slot = (source[index] - mean) * inverse * weight[index] + bias[index];
        }
    }
    out
}

fn reference_run_prefix(model: &SentenceModel, tokens: &[u32]) -> Prefix<'static> {
    let n_embd = model.config.n_embd;
    let mut x = model.embed(tokens, 0);
    let mut keys = Vec::with_capacity(model.blocks.len());
    let mut values = Vec::with_capacity(model.blocks.len());
    for block in &model.blocks {
        let normed = reference_layer_norm(&x, &block.ln1_weight, &block.ln1_bias, n_embd);
        let qkv = reference_linear(
            &normed,
            &block.qkv_weight,
            &block.qkv_bias,
            n_embd,
            3 * n_embd,
        );
        let (q, k, v) = split_qkv(&qkv, tokens.len(), n_embd, &[], &[]);
        let attended = model.attention(&q, &k, &v, tokens.len(), 0);
        let projected = reference_linear(
            &attended,
            &block.proj_weight,
            &block.proj_bias,
            n_embd,
            n_embd,
        );
        for (slot, value) in x.iter_mut().zip(projected) {
            *slot += value;
        }
        reference_feed_forward(model, &mut x, block);
        keys.push(k);
        values.push(v);
    }
    let last = (tokens.len() - 1) * n_embd;
    let last_hidden = Arc::new(HiddenState::new(reference_layer_norm(
        &x[last..],
        &model.final_weight,
        &model.final_bias,
        n_embd,
    )));
    Prefix {
        length: tokens.len(),
        keys: Cow::Owned(keys),
        values: Cow::Owned(values),
        terminal: None,
        last_hidden,
    }
}

fn reference_run_tail_states(
    model: &SentenceModel,
    length: usize,
    prefix_keys: &[Vec<f32>],
    prefix_values: &[Vec<f32>],
    tokens: &[u32],
) -> TailStates {
    // 借用的旧 KV 可能更长，只复制本次续跑可见的位置。
    let prefix_width = length * model.config.n_embd;
    if tokens.is_empty() {
        return TailStates {
            keys: prefix_keys
                .iter()
                .map(|row| row[..prefix_width].to_vec())
                .collect(),
            values: prefix_values
                .iter()
                .map(|row| row[..prefix_width].to_vec())
                .collect(),
            hidden: Vec::new(),
        };
    }
    let n_embd = model.config.n_embd;
    let mut x = model.embed(tokens, length);
    let mut keys_out = Vec::with_capacity(model.blocks.len());
    let mut values_out = Vec::with_capacity(model.blocks.len());
    for (layer, block) in model.blocks.iter().enumerate() {
        let normed = reference_layer_norm(&x, &block.ln1_weight, &block.ln1_bias, n_embd);
        let qkv = reference_linear(
            &normed,
            &block.qkv_weight,
            &block.qkv_bias,
            n_embd,
            3 * n_embd,
        );
        let (q, keys, values) = split_qkv(
            &qkv,
            tokens.len(),
            n_embd,
            &prefix_keys[layer][..prefix_width],
            &prefix_values[layer][..prefix_width],
        );
        let attended = model.attention(&q, &keys, &values, tokens.len(), length);
        let projected = reference_linear(
            &attended,
            &block.proj_weight,
            &block.proj_bias,
            n_embd,
            n_embd,
        );
        for (slot, value) in x.iter_mut().zip(projected) {
            *slot += value;
        }
        reference_feed_forward(model, &mut x, block);
        keys_out.push(keys);
        values_out.push(values);
    }
    let normed = reference_layer_norm(&x, &model.final_weight, &model.final_bias, n_embd);

    TailStates {
        keys: keys_out,
        values: values_out,
        hidden: normed
            .chunks_exact(n_embd)
            .map(|row| Arc::new(HiddenState::new(row.to_vec())))
            .collect(),
    }
}
fn matrix(rows: usize, width: usize, quantized: bool, salt: usize) -> Matrix {
    Matrix {
        inner: if quantized {
            Tensor::Quantized {
                values: (0..rows * width)
                    .map(|i| ((i * 7 + salt) % 23) as i8 - 11)
                    .collect(),
                scales: (0..rows).map(|i| ((i % 7 + 1) as f32) / 128.0).collect(),
            }
        } else {
            Tensor::Float(
                (0..rows * width)
                    .map(|i| (((i * 7 + salt) % 23) as f32 - 11.0) / 64.0)
                    .collect(),
            )
        },
    }
}

fn model(width: usize, layers: usize, quantized: bool) -> SentenceModel {
    let vector = |count, salt| {
        (0..count)
            .map(|i| (((i * 3 + salt) % 19) as f32 - 9.0) / 32.0)
            .collect()
    };
    SentenceModel {
        config: Config {
            vocab: 3,
            n_layer: layers,
            n_head: 1,
            n_embd: width,
            context: 64,
        },
        index: HashMap::new(),
        token: matrix(3, width, quantized, 1),
        position: vec![0.0; 64 * width],
        blocks: (0..layers)
            .map(|layer| Block {
                ln1_weight: vec![1.0; width],
                ln1_bias: vec![0.0; width],
                qkv_weight: matrix(0, width, quantized, 0),
                qkv_bias: Vec::new(),
                proj_weight: matrix(0, width, quantized, 0),
                proj_bias: Vec::new(),
                ln2_weight: vector(width, layer),
                ln2_bias: vector(width, layer + 5),
                fc_weight: matrix(4 * width, width, quantized, layer),
                fc_bias: vector(4 * width, layer + 3),
                out_weight: matrix(width, 4 * width, quantized, layer + 9),
                out_bias: vector(width, layer + 7),
            })
            .collect(),
        final_weight: vec![1.0; width],
        final_bias: vec![0.0; width],
        attribution: String::new(),
        license: String::new(),
    }
}

fn input(width: usize, rows: usize) -> Vec<f32> {
    (0..width * rows)
        .map(|i| {
            if i % 11 == 0 {
                -0.0
            } else {
                ((i * 11 % 31) as f32 - 15.0) / 16.0
            }
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn warmed_feed_forward_keeps_both_buffers() {
    let model = model(8, 1, true);
    let mut x = input(8, 4);
    let mut scratch = FeedForwardScratch::default();
    model.feed_forward(&mut x, &model.blocks[0], &mut scratch);
    let pointers = (scratch.normalized.as_ptr(), scratch.hidden.as_ptr());
    let capacities = (scratch.normalized.capacity(), scratch.hidden.capacity());
    model.feed_forward(&mut x, &model.blocks[0], &mut scratch);
    assert_eq!(
        pointers,
        (scratch.normalized.as_ptr(), scratch.hidden.as_ptr()),
        "同形状前馈仍替换已有缓冲"
    );
    assert_eq!(
        capacities,
        (scratch.normalized.capacity(), scratch.hidden.capacity())
    );
}

#[test]
fn attention_projection_reuses_normalized_buffer() {
    let mut model = model(8, 3, true);
    model.blocks[0].proj_weight = matrix(8, 8, true, 13);
    model.blocks[0].proj_bias = vec![0.0; 8];
    let attended = input(8, 4);
    let mut scratch = FeedForwardScratch::default();
    scratch.normalized.resize(attended.len(), 0.0);
    let pointer = scratch.normalized.as_ptr();
    model.project_attention_into(&attended, &model.blocks[0], &mut scratch);
    let expected = reference_linear(
        &attended,
        &model.blocks[0].proj_weight,
        &model.blocks[0].proj_bias,
        8,
        8,
    );
    assert_eq!(bits(&scratch.normalized), bits(&expected));
    assert!(
        pointer == scratch.normalized.as_ptr(),
        "attention projection should reuse feed-forward scratch"
    );
}

#[test]
fn attention_layer_norm_reuses_normalized_buffer() {
    let model = model(8, 3, true);
    let x = input(8, 4);
    let mut scratch = FeedForwardScratch::default();
    scratch.normalized.resize(x.len(), 0.0);
    let pointer = scratch.normalized.as_ptr();
    model.normalize_attention_into(&x, &model.blocks[0], &mut scratch);
    let expected = reference_layer_norm(
        &x,
        &model.blocks[0].ln1_weight,
        &model.blocks[0].ln1_bias,
        8,
    );
    assert_eq!(bits(&scratch.normalized), bits(&expected));
    assert_eq!(pointer, scratch.normalized.as_ptr());
}

#[test]
fn qkv_projection_reuses_hidden_buffer() {
    let mut model = model(8, 3, true);
    model.blocks[0].qkv_weight = matrix(24, 8, true, 17);
    model.blocks[0].qkv_bias = vec![0.0; 24];
    let normalized = input(8, 4);
    let mut scratch = FeedForwardScratch::default();
    scratch.hidden.resize(normalized.len() * 3, 0.0);
    let pointer = scratch.hidden.as_ptr();
    scratch.normalized.clone_from(&normalized);
    model.qkv_into(&model.blocks[0], &mut scratch);
    let expected = reference_linear(
        &normalized,
        &model.blocks[0].qkv_weight,
        &model.blocks[0].qkv_bias,
        8,
        24,
    );
    assert_eq!(bits(&scratch.hidden[..expected.len()]), bits(&expected));
    assert_eq!(pointer, scratch.hidden.as_ptr());
}

#[test]
fn query_split_reuses_normalized_buffer() {
    let qkv = [
        1.0, -0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let prefix_keys = [21.0, 22.0];
    let prefix_values = [31.0, 32.0];
    let suffix_keys = [41.0, 42.0];
    let suffix_values = [51.0, 52.0];
    let expected = split_qkv_with_suffix(
        &qkv,
        2,
        2,
        &prefix_keys,
        &prefix_values,
        Some((&suffix_keys, &suffix_values)),
    );
    let mut scratch = FeedForwardScratch::default();
    scratch.normalized.resize(4, 0.0);
    let pointer = scratch.normalized.as_ptr();
    let (keys, values) = split_qkv_with_suffix_into(
        &qkv,
        2,
        2,
        &prefix_keys,
        &prefix_values,
        Some((&suffix_keys, &suffix_values)),
        &mut scratch.normalized,
    );
    assert_eq!(bits(&scratch.normalized), bits(&expected.0));
    assert_eq!(bits(&keys), bits(&expected.1));
    assert_eq!(bits(&values), bits(&expected.2));
    assert_eq!(pointer, scratch.normalized.as_ptr());
}

#[test]
fn complete_feed_forward_matches_legacy_bits() {
    for quantized in [false, true] {
        for width in [1, 7, 8, 9, 192, 768] {
            let model = model(width, 3, quantized);
            let mut scratch = FeedForwardScratch::default();
            for rows in [0, 1, 4, 12, 4, 1, 0] {
                let mut old = input(width, rows);
                let mut new = old.clone();
                for block in &model.blocks {
                    reference_feed_forward(&model, &mut old, block);
                    model.feed_forward(&mut new, block, &mut scratch);
                    assert_eq!(
                        bits(&old),
                        bits(&new),
                        "quantized={quantized} width={width} rows={rows}"
                    );
                    assert_eq!(scratch.normalized.len(), rows * width);
                    assert_eq!(scratch.hidden.len(), rows * 4 * width);
                }
            }
        }
    }
}

#[test]
#[ignore = "使用固定真实模型核对前缀与尾部前向的全部 KV 和隐藏状态"]
fn real_model_forward_matches_legacy_bits() {
    let path = std::env::var_os("MSIME_NEURAL_MODEL_DIR").expect("需要固定模型目录");
    let bytes =
        std::fs::read(std::path::Path::new(&path).join("sentence-model.safetensors")).unwrap();
    let model = SentenceModel::load(&bytes).unwrap();
    for context in ["", "甲乙", "合成的测试上文"] {
        let tokens = model.prefix_tokens(context, 8);
        let old = reference_run_prefix(&model, &tokens);
        let new = model.run_prefix(&tokens);
        assert_eq!(bits(&old.last_hidden.values), bits(&new.last_hidden.values));
        for (a, b) in old
            .keys
            .iter()
            .chain(old.values.iter())
            .zip(new.keys.iter().chain(new.values.iter()))
        {
            assert_eq!(bits(a), bits(b));
        }
        for tail in ["", "甲", "甲乙丙", "甲乙丙丁戊己庚辛"] {
            let encoded = model.encode(tail);
            let a = reference_run_tail_states(&model, old.length, &old.keys, &old.values, &encoded);
            let b = model.run_tail_states(new.length, &new.keys, &new.values, None, &encoded);
            for (a, b) in a
                .keys
                .iter()
                .chain(a.values.iter())
                .zip(b.keys.iter().chain(b.values.iter()))
            {
                assert_eq!(bits(a), bits(b));
            }
            assert_eq!(a.hidden.len(), b.hidden.len());
            for (a, b) in a.hidden.iter().zip(&b.hidden) {
                assert_eq!(bits(&a.values), bits(&b.values));
                assert_eq!(
                    bits(&model.distribution(&a.values)),
                    bits(&model.distribution(&b.values))
                );
            }
        }
    }
}

#[inline(never)]
fn old_chain(model: &SentenceModel, x: &[f32]) -> Vec<f32> {
    let mut x = x.to_vec();
    for block in &model.blocks {
        reference_feed_forward(model, &mut x, block);
    }
    x
}

#[inline(never)]
fn new_chain(model: &SentenceModel, x: &[f32]) -> Vec<f32> {
    let mut x = x.to_vec();
    let mut scratch = FeedForwardScratch::default();
    for block in &model.blocks {
        model.feed_forward(&mut x, block, &mut scratch);
    }
    x
}

type Chain = fn(&SentenceModel, &[f32]) -> Vec<f32>;

#[inline(never)]
fn old_layer_norm_chain(model: &SentenceModel, x: &[f32]) -> Vec<f32> {
    let mut x = x.to_vec();
    for block in &model.blocks {
        x = reference_layer_norm(&x, &block.ln1_weight, &block.ln1_bias, model.config.n_embd);
    }
    x
}

#[inline(never)]
fn new_layer_norm_chain(model: &SentenceModel, x: &[f32]) -> Vec<f32> {
    let mut x = x.to_vec();
    let mut scratch = FeedForwardScratch::default();
    for block in &model.blocks {
        model.normalize_attention_into(&x, block, &mut scratch);
        std::mem::swap(&mut x, &mut scratch.normalized);
    }
    x
}

type LayerNormChain = fn(&SentenceModel, &[f32]) -> Vec<f32>;

#[inline(never)]
fn old_qkv_chain(model: &SentenceModel, x: &[f32]) -> Vec<f32> {
    let mut output = Vec::new();
    for block in &model.blocks {
        output = reference_linear(
            x,
            &block.qkv_weight,
            &block.qkv_bias,
            model.config.n_embd,
            3 * model.config.n_embd,
        );
        black_box(&output);
    }
    output
}

#[inline(never)]
fn new_qkv_chain(model: &SentenceModel, x: &[f32]) -> Vec<f32> {
    let mut scratch = FeedForwardScratch {
        normalized: x.to_vec(),
        hidden: Vec::new(),
        kernel: LinearScratch::default(),
    };
    for block in &model.blocks {
        model.qkv_into(block, &mut scratch);
        black_box(&scratch.hidden);
    }
    scratch.hidden
}

type QkvChain = fn(&SentenceModel, &[f32]) -> Vec<f32>;

fn elapsed_qkv(chain: QkvChain, model: &SentenceModel, x: &[f32], repeats: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        drop(black_box(chain(black_box(model), black_box(x))));
    }
    start.elapsed().as_secs_f64()
}

#[test]
#[ignore = "独占运行六层 QKV 配对预算"]
fn six_layer_qkv_paired_budget() {
    let mut failures = Vec::new();
    for quantized in [true, false] {
        for width in [192, 768] {
            let mut model = model(width, 6, quantized);
            for (layer, block) in model.blocks.iter_mut().enumerate() {
                block.qkv_weight = matrix(3 * width, width, quantized, layer);
                block.qkv_bias = vec![0.0; 3 * width];
            }
            for rows in [1, 4, 12] {
                let x = input(width, rows);
                assert_eq!(
                    bits(&old_qkv_chain(&model, &x)),
                    bits(&new_qkv_chain(&model, &x))
                );
                let mut repeats = 1;
                while elapsed_qkv(old_qkv_chain, &model, &x, repeats) < 0.010 {
                    repeats *= 2;
                }
                let mut ratios = Vec::new();
                for pair in 0..40 {
                    let (old, new) = if pair % 2 == 0 {
                        let old = elapsed_qkv(old_qkv_chain, &model, &x, repeats);
                        (old, elapsed_qkv(new_qkv_chain, &model, &x, repeats))
                    } else {
                        let new = elapsed_qkv(new_qkv_chain, &model, &x, repeats);
                        (elapsed_qkv(old_qkv_chain, &model, &x, repeats), new)
                    };
                    ratios.push(new / old);
                }
                ratios.sort_by(f64::total_cmp);
                let ratio = ratios[19];
                let budget = if quantized && width == 192 && rows == 1 {
                    0.99
                } else {
                    1.03
                };
                println!(
                    "quantized={quantized} width={width} rows={rows} repeats={repeats} paired_ratio_p50={ratio:.4} budget={budget:.2}"
                );
                if ratio >= budget {
                    failures.push((quantized, width, rows, ratio, budget));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "paired QKV budget failures: {failures:?}"
    );
}

#[inline(never)]
fn old_query_split_chain(qkv: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let mut query = Vec::new();
    for _ in 0..6 {
        let (current, keys, values) = split_qkv(qkv, rows, width, &[], &[]);
        black_box((&keys, &values));
        query = current;
    }
    query
}

#[inline(never)]
fn new_query_split_chain(qkv: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let mut query = Vec::new();
    for _ in 0..6 {
        let (keys, values) =
            split_qkv_with_suffix_into(qkv, rows, width, &[], &[], None, &mut query);
        black_box((&keys, &values));
    }
    query
}

type QuerySplitChain = fn(&[f32], usize, usize) -> Vec<f32>;

fn elapsed_query_split(
    chain: QuerySplitChain,
    qkv: &[f32],
    rows: usize,
    width: usize,
    repeats: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        drop(black_box(chain(black_box(qkv), rows, width)));
    }
    start.elapsed().as_secs_f64()
}

#[test]
#[ignore = "独占运行六层 query 拆分配对预算"]
fn six_layer_query_split_paired_budget() {
    let mut failures = Vec::new();
    for quantized in [true, false] {
        for width in [192, 768] {
            for rows in [1, 4, 12] {
                let mut qkv = input(width * 3, rows);
                if quantized {
                    qkv[0] = -qkv[0];
                }
                assert_eq!(
                    bits(&old_query_split_chain(&qkv, rows, width)),
                    bits(&new_query_split_chain(&qkv, rows, width))
                );
                let mut repeats = 1;
                while elapsed_query_split(old_query_split_chain, &qkv, rows, width, repeats) < 0.010
                {
                    repeats *= 2;
                }
                let mut ratios = Vec::new();
                for pair in 0..40 {
                    let (old, new) = if pair % 2 == 0 {
                        let old =
                            elapsed_query_split(old_query_split_chain, &qkv, rows, width, repeats);
                        (
                            old,
                            elapsed_query_split(new_query_split_chain, &qkv, rows, width, repeats),
                        )
                    } else {
                        let new =
                            elapsed_query_split(new_query_split_chain, &qkv, rows, width, repeats);
                        (
                            elapsed_query_split(old_query_split_chain, &qkv, rows, width, repeats),
                            new,
                        )
                    };
                    ratios.push(new / old);
                }
                ratios.sort_by(f64::total_cmp);
                let ratio = ratios[19];
                let budget = if quantized && width == 192 && rows == 1 {
                    0.99
                } else {
                    1.03
                };
                println!(
                    "quantized={quantized} width={width} rows={rows} repeats={repeats} paired_ratio_p50={ratio:.4} budget={budget:.2}"
                );
                if ratio >= budget {
                    failures.push((quantized, width, rows, ratio, budget));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "paired query split budget failures: {failures:?}"
    );
}

fn elapsed_layer_norm(
    chain: LayerNormChain,
    model: &SentenceModel,
    x: &[f32],
    repeats: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        drop(black_box(chain(black_box(model), black_box(x))));
    }
    start.elapsed().as_secs_f64()
}

#[test]
#[ignore = "独占运行六层 attention 前置归一化配对预算"]
fn six_layer_attention_layer_norm_paired_budget() {
    let mut failures = Vec::new();
    for quantized in [true, false] {
        for width in [192, 768] {
            let model = model(width, 6, quantized);
            for rows in [1, 4, 12] {
                let x = input(width, rows);
                assert_eq!(
                    bits(&old_layer_norm_chain(&model, &x)),
                    bits(&new_layer_norm_chain(&model, &x))
                );
                let mut repeats = 1;
                while elapsed_layer_norm(old_layer_norm_chain, &model, &x, repeats) < 0.010 {
                    repeats *= 2;
                }
                let mut ratios = Vec::new();
                for pair in 0..40 {
                    let (old, new) = if pair % 2 == 0 {
                        let old = elapsed_layer_norm(old_layer_norm_chain, &model, &x, repeats);
                        (
                            old,
                            elapsed_layer_norm(new_layer_norm_chain, &model, &x, repeats),
                        )
                    } else {
                        let new = elapsed_layer_norm(new_layer_norm_chain, &model, &x, repeats);
                        (
                            elapsed_layer_norm(old_layer_norm_chain, &model, &x, repeats),
                            new,
                        )
                    };
                    ratios.push(new / old);
                }
                ratios.sort_by(f64::total_cmp);
                let ratio = ratios[19];
                let budget = if quantized && width == 192 && rows == 1 {
                    0.99
                } else {
                    1.03
                };
                println!(
                    "quantized={quantized} width={width} rows={rows} repeats={repeats} paired_ratio_p50={ratio:.4} budget={budget:.2}"
                );
                if ratio >= budget {
                    failures.push((quantized, width, rows, ratio, budget));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "paired layer norm budget failures: {failures:?}"
    );
}

fn elapsed(chain: Chain, model: &SentenceModel, x: &[f32], repeats: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        drop(black_box(chain(black_box(model), black_box(x))));
    }
    start.elapsed().as_secs_f64()
}

#[test]
#[ignore = "独占运行完整六层前馈链配对预算"]
fn six_layer_feed_forward_paired_budget() {
    let mut failures = Vec::new();
    for quantized in [true, false] {
        for width in [192, 768] {
            let model = model(width, 6, quantized);
            for rows in [1, 4, 12] {
                let x = input(width, rows);
                assert_eq!(bits(&old_chain(&model, &x)), bits(&new_chain(&model, &x)));
                let mut repeats = 1;
                while elapsed(old_chain, &model, &x, repeats) < 0.010 {
                    repeats *= 2;
                }
                let mut ratios = Vec::new();
                for pair in 0..40 {
                    let (old, new) = if pair % 2 == 0 {
                        let old = elapsed(old_chain, &model, &x, repeats);
                        (old, elapsed(new_chain, &model, &x, repeats))
                    } else {
                        let new = elapsed(new_chain, &model, &x, repeats);
                        (elapsed(old_chain, &model, &x, repeats), new)
                    };
                    ratios.push(new / old);
                }
                ratios.sort_by(f64::total_cmp);
                let ratio = ratios[19];
                let budget = if quantized && width == 192 && rows == 1 {
                    0.99
                } else {
                    1.03
                };
                println!("quantized={quantized} width={width} rows={rows} repeats={repeats} paired_ratio_p50={ratio:.4} budget={budget:.2}");
                if ratio >= budget {
                    failures.push((quantized, width, rows, ratio, budget));
                }
            }
        }
    }
    assert!(failures.is_empty(), "完整前馈预算失败：{failures:?}");
}
