//! A tiny model laid out from `docs/format.md`, shared by the Rust tests and by `examples/write_fixture.rs`, which hands the same bytes to the C smoke test.
//!
//! Built here rather than checked in, as `reference/tests/model_file.rs` does, so that a change to the format shows up as a failing test instead of a stale binary.

use std::collections::BTreeMap;

const VOCAB: usize = 8;
const LAYERS: usize = 1;
const HEADS: usize = 2;
const EMBED: usize = 4;
const CONTEXT: usize = 8;

/// The five characters this vocabulary knows, after the three reserved tokens.
const CHARACTERS: [char; 5] = ['你', '好', '吗', '我', '很'];

pub const LICENSE: &str = "Apache-2.0";
pub const ATTRIBUTION: &str = "test fixture";

/// Deterministic values in a small range, so a forward pass stays finite and comparable.
fn ramp(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((index + seed) % 17) as f32 - 8.0) / 32.0)
        .collect()
}

/// Every tensor the format requires, in float32.
fn tensors() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut tensors = vec![
        (
            "tok.weight".to_owned(),
            vec![VOCAB, EMBED],
            ramp(VOCAB * EMBED, 1),
        ),
        (
            "pos.weight".to_owned(),
            vec![CONTEXT, EMBED],
            ramp(CONTEXT * EMBED, 2),
        ),
    ];
    for layer in 0..LAYERS {
        let name = |suffix: &str| format!("blocks.{layer}.{suffix}");
        tensors.extend([
            (name("ln1.weight"), vec![EMBED], vec![1.0; EMBED]),
            (name("ln1.bias"), vec![EMBED], vec![0.0; EMBED]),
            (
                name("qkv.weight"),
                vec![3 * EMBED, EMBED],
                ramp(3 * EMBED * EMBED, 3),
            ),
            (name("qkv.bias"), vec![3 * EMBED], ramp(3 * EMBED, 4)),
            (
                name("proj.weight"),
                vec![EMBED, EMBED],
                ramp(EMBED * EMBED, 5),
            ),
            (name("proj.bias"), vec![EMBED], ramp(EMBED, 6)),
            (name("ln2.weight"), vec![EMBED], vec![1.0; EMBED]),
            (name("ln2.bias"), vec![EMBED], vec![0.0; EMBED]),
            (
                name("fc.weight"),
                vec![4 * EMBED, EMBED],
                ramp(4 * EMBED * EMBED, 7),
            ),
            (name("fc.bias"), vec![4 * EMBED], ramp(4 * EMBED, 8)),
            (
                name("out.weight"),
                vec![EMBED, 4 * EMBED],
                ramp(4 * EMBED * EMBED, 9),
            ),
            (name("out.bias"), vec![EMBED], ramp(EMBED, 10)),
        ]);
    }
    tensors.extend([
        ("ln_f.weight".to_owned(), vec![EMBED], vec![1.0; EMBED]),
        ("ln_f.bias".to_owned(), vec![EMBED], ramp(EMBED, 11)),
    ]);
    tensors
}

/// The bytes of a complete, loadable float32 model file.
pub fn model_bytes() -> Vec<u8> {
    let config = format!(
        r#"{{"vocab":{VOCAB},"n_layer":{LAYERS},"n_head":{HEADS},"n_embd":{EMBED},"context":{CONTEXT},"dropout":0.0}}"#
    );
    let tokens: Vec<String> = ["<pad>", "<unk>", "<bos>"]
        .iter()
        .map(|token| (*token).to_owned())
        .chain(CHARACTERS.iter().map(char::to_string))
        .collect();
    let metadata = BTreeMap::from([
        ("format", "chinese-ime-lm".to_owned()),
        ("version", "1".to_owned()),
        ("precision", "f32".to_owned()),
        ("config", config),
        ("vocab", serde_json::to_string(&tokens).expect("vocabulary")),
        ("attribution", ATTRIBUTION.to_owned()),
        ("license", LICENSE.to_owned()),
    ]);

    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::to_value(&metadata).expect("metadata"),
    );
    let mut data = Vec::new();
    for (name, shape, values) in tensors() {
        let start = data.len();
        data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        header.insert(
            name,
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    bytes
}
