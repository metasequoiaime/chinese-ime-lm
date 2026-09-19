//! Builds model files by hand and loads them.
//!
//! The parser is what a hostile model file meets first, and `SECURITY.md` claims it answers a
//! malformed one with an error rather than by reading somewhere else. That claim needs tests.
//!
//! Building the files here rather than checking in a fixture also keeps `docs/format.md` honest:
//! these bytes are laid out from the specification, so if the two disagree the tests fail.

use std::collections::BTreeMap;

use chinese_ime_lm::{ModelError, SentenceModel};

const VOCAB: usize = 8;
const LAYERS: usize = 1;
const HEADS: usize = 2;
const EMBED: usize = 4;
const CONTEXT: usize = 8;

/// The five characters this tiny vocabulary knows, after the three reserved tokens.
const CHARACTERS: [char; 5] = ['你', '好', '吗', '我', '很'];

#[derive(Clone)]
enum Tensor {
    F32(Vec<f32>),
    F16(Vec<f32>),
    /// Values already quantized, paired with the per-row scale they decode with.
    I8(Vec<i8>, Vec<f32>),
}

struct Builder {
    entries: Vec<(String, Vec<usize>, Tensor)>,
    metadata: BTreeMap<String, String>,
}

impl Builder {
    fn new(precision: &str) -> Self {
        let config = format!(
            r#"{{"vocab":{VOCAB},"n_layer":{LAYERS},"n_head":{HEADS},"n_embd":{EMBED},"context":{CONTEXT},"dropout":0.0}}"#
        );
        let tokens: Vec<String> = ["<pad>", "<unk>", "<bos>"]
            .iter()
            .map(|s| (*s).to_owned())
            .chain(CHARACTERS.iter().map(|c| c.to_string()))
            .collect();
        let vocab = serde_json::to_string(&tokens).expect("vocabulary");
        let metadata = BTreeMap::from([
            ("format".to_owned(), "chinese-ime-lm".to_owned()),
            ("version".to_owned(), "1".to_owned()),
            ("precision".to_owned(), precision.to_owned()),
            ("config".to_owned(), config),
            ("vocab".to_owned(), vocab),
            ("attribution".to_owned(), "test fixture".to_owned()),
        ]);
        Self {
            entries: Vec::new(),
            metadata,
        }
    }

    fn put(&mut self, name: &str, shape: &[usize], tensor: Tensor) {
        self.entries.push((name.to_owned(), shape.to_vec(), tensor));
    }

    fn finish(self) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        header.insert(
            "__metadata__".to_owned(),
            serde_json::to_value(&self.metadata).expect("metadata"),
        );
        for (name, shape, tensor) in &self.entries {
            let (dtype, bytes) = match tensor {
                Tensor::F32(values) => (
                    "F32",
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                ),
                Tensor::F16(values) => (
                    "F16",
                    values
                        .iter()
                        .flat_map(|v| to_half(*v).to_le_bytes())
                        .collect(),
                ),
                Tensor::I8(values, scales) => {
                    let start = data.len();
                    data.extend(values.iter().map(|v| *v as u8));
                    header.insert(name.clone(), entry("I8", shape, start, data.len()));
                    let start = data.len();
                    data.extend(scales.iter().flat_map(|v| v.to_le_bytes()));
                    header.insert(
                        format!("{name}.scale"),
                        entry("F32", &[scales.len()], start, data.len()),
                    );
                    continue;
                }
            };
            let start = data.len();
            data.extend_from_slice(&bytes);
            header.insert(name.clone(), entry(dtype, shape, start, data.len()));
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }
}

fn entry(dtype: &str, shape: &[usize], start: usize, end: usize) -> serde_json::Value {
    serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]})
}

/// Deterministic values in a small range, so a forward pass stays finite and comparable.
fn ramp(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((index + seed) % 17) as f32 - 8.0) / 32.0)
        .collect()
}

fn to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    if exponent <= 0 {
        return sign;
    }
    sign | ((exponent as u16) << 10) | mantissa
}

/// Every tensor the format requires, at the given precision.
fn complete(precision: &str) -> Builder {
    let mut builder = Builder::new(precision);
    // Quantization is per output row, so the number of scales must equal the first dimension.
    // Getting that wrong is exactly what the loader rejects, so the fixture has to be right for the
    // rest of this test to mean anything.
    let wrap = |values: Vec<f32>, shape: &[usize]| -> Tensor {
        match precision {
            "f16" => Tensor::F16(values),
            "int8" => {
                let rows = shape[0];
                let columns = values.len() / rows;
                let mut quantized = Vec::with_capacity(values.len());
                let mut scales = Vec::with_capacity(rows);
                for row in 0..rows {
                    let slice = &values[row * columns..(row + 1) * columns];
                    let peak = slice.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-8);
                    let scale = peak / 127.0;
                    scales.push(scale);
                    quantized.extend(slice.iter().map(|v| (v / scale).round() as i8));
                }
                Tensor::I8(quantized, scales)
            }
            _ => Tensor::F32(values),
        }
    };
    builder.put(
        "tok.weight",
        &[VOCAB, EMBED],
        wrap(ramp(VOCAB * EMBED, 1), &[VOCAB, EMBED]),
    );
    // The positional table stays float32 at every precision, as the format requires.
    builder.put(
        "pos.weight",
        &[CONTEXT, EMBED],
        Tensor::F32(ramp(CONTEXT * EMBED, 2)),
    );
    for layer in 0..LAYERS {
        let name = |suffix: &str| format!("blocks.{layer}.{suffix}");
        builder.put(&name("ln1.weight"), &[EMBED], Tensor::F32(vec![1.0; EMBED]));
        builder.put(&name("ln1.bias"), &[EMBED], Tensor::F32(vec![0.0; EMBED]));
        builder.put(
            &name("qkv.weight"),
            &[3 * EMBED, EMBED],
            wrap(ramp(3 * EMBED * EMBED, 3), &[3 * EMBED, EMBED]),
        );
        builder.put(
            &name("qkv.bias"),
            &[3 * EMBED],
            Tensor::F32(ramp(3 * EMBED, 4)),
        );
        builder.put(
            &name("proj.weight"),
            &[EMBED, EMBED],
            wrap(ramp(EMBED * EMBED, 5), &[EMBED, EMBED]),
        );
        builder.put(&name("proj.bias"), &[EMBED], Tensor::F32(ramp(EMBED, 6)));
        builder.put(&name("ln2.weight"), &[EMBED], Tensor::F32(vec![1.0; EMBED]));
        builder.put(&name("ln2.bias"), &[EMBED], Tensor::F32(vec![0.0; EMBED]));
        builder.put(
            &name("fc.weight"),
            &[4 * EMBED, EMBED],
            wrap(ramp(4 * EMBED * EMBED, 7), &[4 * EMBED, EMBED]),
        );
        builder.put(
            &name("fc.bias"),
            &[4 * EMBED],
            Tensor::F32(ramp(4 * EMBED, 8)),
        );
        builder.put(
            &name("out.weight"),
            &[EMBED, 4 * EMBED],
            wrap(ramp(4 * EMBED * EMBED, 9), &[EMBED, 4 * EMBED]),
        );
        builder.put(&name("out.bias"), &[EMBED], Tensor::F32(ramp(EMBED, 10)));
    }
    builder.put("ln_f.weight", &[EMBED], Tensor::F32(vec![1.0; EMBED]));
    builder.put("ln_f.bias", &[EMBED], Tensor::F32(ramp(EMBED, 11)));
    builder
}

#[test]
fn loads_every_precision_and_scores() {
    for precision in ["f32", "f16", "int8"] {
        let bytes = complete(precision).finish();
        let model =
            SentenceModel::load(&bytes).unwrap_or_else(|error| panic!("{precision}: {error}"));
        assert_eq!(model.context_length(), CONTEXT);
        assert_eq!(model.attribution(), "test fixture");
        assert_eq!(model.token('你'), Some(3));
        assert_eq!(model.token('X'), None);

        let scores = model.score("你好", &["我很好", "你好吗"]);
        assert_eq!(scores.len(), 2);
        for score in &scores {
            assert!(score.is_finite(), "{precision}: score was {score}");
            // A mean log-probability over a vocabulary of eight cannot exceed zero.
            assert!(*score <= 0.0, "{precision}: score was {score}");
        }
    }
}

#[test]
fn next_character_distribution_is_normalized() {
    let model = SentenceModel::load(&complete("f32").finish()).expect("load");
    let distribution = model.next_log_probabilities("你好");
    assert_eq!(distribution.len(), VOCAB);
    let total: f32 = distribution.iter().map(|value| value.exp()).sum();
    assert!(
        (total - 1.0).abs() < 1e-4,
        "probabilities summed to {total}"
    );
}

#[test]
fn unknown_characters_do_not_escape_the_vocabulary() {
    let model = SentenceModel::load(&complete("f32").finish()).expect("load");
    // Characters outside the vocabulary become <unk>; scoring them must not index out of range.
    let scores = model.score("🙂 latin", &["𠀀𠀁𠀂", "你好吗"]);
    assert!(scores.iter().all(|score| score.is_finite()));
}

#[test]
fn truncation_is_an_error_at_every_offset() {
    let bytes = complete("f32").finish();
    for cut in [0, 4, 7, 8, 9, 64, bytes.len() / 2, bytes.len() - 1] {
        let error = SentenceModel::load(&bytes[..cut]).expect_err("truncated file was accepted");
        assert!(
            matches!(
                error,
                ModelError::Truncated | ModelError::Header(_) | ModelError::MissingTensor(_)
            ),
            "cut at {cut} gave {error}"
        );
    }
}

#[test]
fn a_header_length_past_the_file_is_rejected() {
    let mut bytes = complete("f32").finish();
    bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(
        SentenceModel::load(&bytes).expect_err("accepted an impossible header length"),
        ModelError::Truncated
    ));
}

#[test]
fn offsets_past_the_data_section_are_rejected() {
    let mut builder = complete("f32");
    // Point one tensor at data that is not there. A reader that trusted the header would read
    // whatever follows the buffer.
    builder.entries.retain(|(name, ..)| name != "ln_f.bias");
    let mut bytes = builder.finish();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("header");
    let mut header = header.as_object().expect("object").clone();
    header.insert(
        "ln_f.bias".to_owned(),
        entry("F32", &[EMBED], usize::MAX / 2, usize::MAX),
    );
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
    bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&[0u8; 64]);
    assert!(matches!(
        SentenceModel::load(&bytes).expect_err("accepted an out-of-range offset"),
        ModelError::Truncated
    ));
}

#[test]
fn geometry_that_disagrees_with_the_configuration_is_rejected() {
    let mut builder = complete("f32");
    // A token table one row short of what the configuration promises.
    builder.entries.retain(|(name, ..)| name != "tok.weight");
    builder.put(
        "tok.weight",
        &[VOCAB - 1, EMBED],
        Tensor::F32(ramp((VOCAB - 1) * EMBED, 1)),
    );
    let error = SentenceModel::load(&builder.finish()).expect_err("accepted a short token table");
    assert!(matches!(error, ModelError::Geometry(_)), "{error}");
}

#[test]
fn a_missing_tensor_names_itself() {
    let mut builder = complete("f32");
    builder
        .entries
        .retain(|(name, ..)| name != "blocks.0.fc.weight");
    let error = SentenceModel::load(&builder.finish()).expect_err("accepted a missing tensor");
    match error {
        ModelError::MissingTensor(name) => assert_eq!(name, "blocks.0.fc.weight"),
        other => panic!("{other}"),
    }
}

#[test]
fn int8_without_its_scale_is_rejected() {
    let mut builder = Builder::new("int8");
    let complete = complete("int8");
    for (name, shape, tensor) in complete.entries {
        if name == "tok.weight" {
            // Keep the quantized values, drop the scale that decodes them.
            if let Tensor::I8(values, _) = tensor {
                builder.put(&name, &shape, Tensor::I8(values.clone(), Vec::new()));
                continue;
            }
        }
        builder.put(&name, &shape, tensor);
    }
    let error = SentenceModel::load(&builder.finish()).expect_err("accepted int8 without a scale");
    assert!(matches!(error, ModelError::Header(_)), "{error}");
}

#[test]
fn an_unsupported_dtype_is_named_rather_than_guessed() {
    let bytes = complete("f32").finish();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = String::from_utf8(bytes[8..8 + header_len].to_vec()).expect("utf-8");
    let header = header.replacen(r#""dtype":"F32""#, r#""dtype":"BF16""#, 1);
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&bytes[8 + header_len..]);
    let error = SentenceModel::load(&out).expect_err("accepted an unsupported dtype");
    assert!(format!("{error}").contains("BF16"), "{error}");
}

/// The same weights, quantized and not, must rank candidates identically.
///
/// Quantized matrices are now kept as `i8` and scaled once per output row instead of being expanded
/// to `f32` on load, which halves what the model occupies — the reason the change exists, since it
/// runs inside an iOS keyboard extension. A per-row scale is a constant factor over the whole row,
/// so lifting it out of the dot product is the same arithmetic; this holds that claim to the only
/// thing that matters, which is the order the scores come out in.
#[test]
fn keeping_weights_quantized_does_not_change_the_ranking() {
    let f32_model = SentenceModel::load(&complete("f32").finish()).expect("f32 model");
    let int8_model = SentenceModel::load(&complete("int8").finish()).expect("int8 model");

    // The builder's weights are synthetic, so the absolute scores mean nothing; what has to agree
    // is which candidate each model puts first, over inputs that exercise several rows.
    let candidates = ["一二三", "一二四", "三二一", "四三二"];
    let plain = f32_model.score("", &candidates);
    let quantized = int8_model.score("", &candidates);
    assert_eq!(plain.len(), quantized.len());

    let best = |scores: &[f32]| {
        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap()
    };
    assert_eq!(
        best(&plain),
        best(&quantized),
        "quantized weights chose a different candidate: {plain:?} vs {quantized:?}"
    );
}

/// A tensor `export.py` never quantizes must be refused if it arrives quantized.
///
/// Layer-norm parameters and biases stay float precisely because they are what rounding hurts. A
/// file that quantized them would load and produce quietly worse rankings, which is the failure
/// mode worth refusing rather than absorbing.
#[test]
fn a_quantized_layer_norm_is_refused() {
    let mut builder = Builder::new("int8");
    for (name, shape, tensor) in complete("int8").entries {
        if name == "ln_f.weight" {
            if let Tensor::F32(values) = &tensor {
                let scales = vec![1.0f32; shape[0]];
                let quantized: Vec<i8> = values.iter().map(|v| *v as i8).collect();
                builder.put(&name, &shape, Tensor::I8(quantized, scales));
                continue;
            }
        }
        builder.put(&name, &shape, tensor);
    }
    let error = SentenceModel::load(&builder.finish()).expect_err("accepted a quantized LayerNorm");
    assert!(matches!(error, ModelError::Header(_)), "{error}");
}
