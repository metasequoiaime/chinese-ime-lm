use super::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

// 固定 926915e73bfafbc2545574c1203a98f0f4f849ba 的完整嵌入路径，保持临时行申请与析构。
fn legacy_embed(model: &SentenceModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    FORWARD_POSITIONS.with(|count| count.set(count.get() + tokens.len()));
    let n_embd = model.config.n_embd;
    let mut x = vec![0.0; tokens.len() * n_embd];
    for (step, &token) in tokens.iter().enumerate() {
        let token = (token as usize).min(model.config.vocab - 1);
        let position = (offset + step).min(model.config.context - 1);
        let row = &mut x[step * n_embd..(step + 1) * n_embd];
        let span = token * n_embd..(token + 1) * n_embd;
        let token_row: Vec<f32> = match &model.token.inner {
            Tensor::Float(values) => values[span].to_vec(),
            Tensor::Quantized { values, scales } => {
                let scale = scales[token];
                values[span].iter().map(|&v| f32::from(v) * scale).collect()
            }
        };
        let position_row = &model.position[position * n_embd..(position + 1) * n_embd];
        for ((out, &t), &p) in row.iter_mut().zip(&token_row).zip(position_row) {
            *out = t + p;
        }
    }
    x
}

fn fixture(width: usize, quantized: bool) -> SentenceModel {
    let config = Config {
        vocab: 17,
        n_layer: 0,
        n_head: 1,
        n_embd: width,
        context: 64,
    };
    let integers: Vec<i8> = (0..config.vocab * width)
        .map(|i| ((i * 31 % 255) as i16 - 127) as i8)
        .collect();
    let inner = if quantized {
        Tensor::Quantized {
            values: integers,
            scales: (0..config.vocab).map(|i| (i + 1) as f32 / 1301.0).collect(),
        }
    } else {
        Tensor::Float(integers.iter().map(|&v| f32::from(v) / 1301.0).collect())
    };
    SentenceModel {
        position: (0..config.context * width)
            .map(|i| ((i * 17 % 127) as f32 - 63.0) / 37.0)
            .collect(),
        final_weight: vec![1.0; width],
        final_bias: vec![0.0; width],
        config,
        index: HashMap::new(),
        token: Matrix { inner },
        blocks: Vec::new(),
        attribution: String::new(),
        license: String::new(),
    }
}

fn assert_same(model: &SentenceModel, tokens: &[u32], offset: usize) {
    let old = legacy_embed(model, tokens, offset);
    let new = model.embed(tokens, offset);
    assert_eq!(
        old.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        new.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn embedding_matches_frozen_bits() {
    for width in [1, 7, 8, 9, 192, 768] {
        for quantized in [false, true] {
            let model = fixture(width, quantized);
            for rows in [0, 1, 4, 12, 64] {
                let mut tokens: Vec<_> =
                    (0..rows).map(|i| (i % model.config.vocab) as u32).collect();
                if let Some(last) = tokens.last_mut() {
                    *last = u32::MAX;
                }
                for offset in [0, 19, 63, 100] {
                    assert_same(&model, &tokens, offset);
                }
            }
        }
    }
}

#[inline(never)]
fn old(model: &SentenceModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    legacy_embed(model, tokens, offset)
}
#[inline(never)]
fn new(model: &SentenceModel, tokens: &[u32], offset: usize) -> Vec<f32> {
    model.embed(tokens, offset)
}

type Embed = fn(&SentenceModel, &[u32], usize) -> Vec<f32>;
fn timed(f: Embed, model: &SentenceModel, tokens: &[u32], count: usize) -> Duration {
    let started = Instant::now();
    for _ in 0..count {
        drop(black_box(f(
            black_box(model),
            black_box(tokens),
            black_box(19),
        )));
    }
    started.elapsed()
}

#[test]
#[ignore = "手动 release 配对预算，普通测试不依赖墙钟"]
fn embedding_paired_budget() {
    let mut failures = Vec::new();
    for width in [192, 768] {
        for quantized in [false, true] {
            let model = fixture(width, quantized);
            for rows in [1, 4, 12, 64] {
                let tokens: Vec<_> = (0..rows).map(|i| (i % model.config.vocab) as u32).collect();
                assert_same(&model, &tokens, 19);
                let mut count = 1;
                // 三次校准都达到时长，避免单次被调度出去便过早停止。
                while (0..3)
                    .map(|_| timed(old, &model, &tokens, count))
                    .min()
                    .unwrap()
                    < Duration::from_millis(5)
                {
                    count *= 2;
                }
                let mut ratios = Vec::new();
                let mut old_us = Vec::new();
                let mut new_us = Vec::new();
                for pair in 0..21 {
                    let (a, b) = if pair % 2 == 0 {
                        (
                            timed(old, &model, &tokens, count),
                            timed(new, &model, &tokens, count),
                        )
                    } else {
                        let b = timed(new, &model, &tokens, count);
                        (timed(old, &model, &tokens, count), b)
                    };
                    eprintln!("pair width={width} quantized={quantized} rows={rows} pair={pair} old_ns={} new_ns={} count={count}", a.as_nanos(), b.as_nanos());
                    ratios.push(b.as_secs_f64() / a.as_secs_f64());
                    old_us.push(a.as_secs_f64() * 1e6 / count as f64);
                    new_us.push(b.as_secs_f64() * 1e6 / count as f64);
                }
                ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
                old_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                new_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                eprintln!("width={width} quantized={quantized} rows={rows} old_us={:.6} new_us={:.6} ratio={:.6} p10={:.6} p90={:.6} count={count}",old_us[10],new_us[10],ratios[10],ratios[2],ratios[18]);
                if ratios[10] >= 0.95 {
                    failures.push((width, quantized, rows, ratios[10]));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "嵌入完整入口未达到预设 5% 收益: {failures:?}"
    );
}

#[test]
#[ignore = "需要显式指定固定真实模型"]
fn real_embedding_matches_frozen_bits() {
    let path = std::env::var_os("MSIME_NEURAL_MODEL_DIR").expect("需要固定模型目录");
    let bytes =
        std::fs::read(std::path::Path::new(&path).join("sentence-model.safetensors")).unwrap();
    let model = SentenceModel::load(&bytes).unwrap();
    for rows in [0, 1, 4, 12, 64] {
        let mut tokens: Vec<_> = (0..rows)
            .map(|i| ((i * 701) % model.config.vocab) as u32)
            .collect();
        if let Some(last) = tokens.last_mut() {
            *last = u32::MAX;
        }
        for offset in [0, 19, 63, 100] {
            assert_same(&model, &tokens, offset);
        }
    }
}
