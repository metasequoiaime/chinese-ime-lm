//! The batched paths against the one-at-a-time code they replaced, bit for bit.
//!
//! Every change in the batched forward is a change of schedule: which rows go through a matrix
//! product together, and in which order positions are computed. None of it may change a single
//! score, because a score that moves in its last bit can flip a pick that sits on a tie, and every
//! published accuracy number for this crate assumes the picks it measured. So these tests compare
//! bits, not tolerances, and they compare the amount of work too: a batch that ran a position twice
//! or skipped one would still produce the right numbers here and the wrong latency in a host.

use super::*;

/// A small deterministic stream, so fixtures are not built from arithmetic progressions whose
/// regularity could hide an indexing mistake.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

const ALPHABET: [char; 12] = [
    '你', '好', '我', '很', '吗', '他', '们', '在', '开', '会', '是', '的',
];

fn matrix(stream: &mut Stream, rows: usize, inputs: usize, quantized: bool) -> Matrix {
    Matrix {
        inner: if quantized {
            Tensor::Quantized {
                values: (0..rows * inputs)
                    .map(|_| (stream.next() % 255) as u8 as i8)
                    .collect(),
                scales: (0..rows)
                    .map(|_| (stream.below(15) + 1) as f32 / 512.0)
                    .collect(),
            }
        } else {
            Tensor::Float((0..rows * inputs).map(|_| stream.unit() / 4.0).collect())
        },
    }
}

/// A model with every dimension chosen to miss the kernel's block sizes: the width is not a
/// multiple of eight, and the vocabulary is not a multiple of anything.
fn model(width: usize, heads: usize, layers: usize, quantized: bool) -> Arc<SentenceModel> {
    let mut stream = Stream(0x9e37_79b9_7f4a_7c15 ^ (width * 31 + layers) as u64);
    let vocab = 3 + ALPHABET.len() + 2;
    let context = 24;
    let vector = |count: usize, stream: &mut Stream| -> Vec<f32> {
        (0..count).map(|_| stream.unit() / 2.0).collect()
    };
    let blocks = (0..layers)
        .map(|_| Block {
            ln1_weight: vector(width, &mut stream).iter().map(|v| 1.0 + v).collect(),
            ln1_bias: vector(width, &mut stream),
            qkv_weight: matrix(&mut stream, 3 * width, width, quantized),
            qkv_bias: vector(3 * width, &mut stream),
            proj_weight: matrix(&mut stream, width, width, quantized),
            proj_bias: vector(width, &mut stream),
            ln2_weight: vector(width, &mut stream).iter().map(|v| 1.0 + v).collect(),
            ln2_bias: vector(width, &mut stream),
            fc_weight: matrix(&mut stream, 4 * width, width, quantized),
            fc_bias: vector(4 * width, &mut stream),
            out_weight: matrix(&mut stream, width, 4 * width, quantized),
            out_bias: vector(width, &mut stream),
        })
        .collect();
    Arc::new(SentenceModel {
        config: Config {
            vocab,
            n_layer: layers,
            n_head: heads,
            n_embd: width,
            context,
        },
        index: ALPHABET
            .iter()
            .enumerate()
            .map(|(index, &character)| (character, index as u32 + 3))
            .collect(),
        token: matrix(&mut stream, vocab, width, quantized),
        position: vector(context * width, &mut stream),
        blocks,
        final_weight: vector(width, &mut stream).iter().map(|v| 1.0 + v).collect(),
        final_bias: vector(width, &mut stream),
        attribution: String::new(),
        license: String::new(),
    })
}

fn models() -> Vec<Arc<SentenceModel>> {
    vec![
        model(20, 4, 2, true),
        model(20, 5, 1, false),
        model(16, 2, 2, true),
    ]
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A candidate drawn from a few stems, so that candidates share prefixes the way an engine's do,
/// with an occasional character the model does not know.
fn candidate(stream: &mut Stream, stems: &[String]) -> String {
    let mut text: String = stems[stream.below(stems.len())]
        .chars()
        .take(stream.below(9))
        .collect();
    for _ in 0..stream.below(4) {
        text.push(match stream.below(13) {
            12 => 'X',
            index => ALPHABET[index],
        });
    }
    text
}

fn stems(stream: &mut Stream) -> Vec<String> {
    (0..3)
        .map(|_| {
            (0..10)
                .map(|_| ALPHABET[stream.below(ALPHABET.len())])
                .collect()
        })
        .collect()
}

#[test]
fn blocked_linear_matches_the_row_at_a_time_loop() {
    let mut stream = Stream(7);
    for quantized in [false, true] {
        for inputs in [1, 7, 8, 12, 20, 33] {
            for outputs in [1, 5, 17] {
                let weights = matrix(&mut stream, outputs, inputs, quantized);
                let bias: Vec<f32> = (0..outputs).map(|_| stream.unit()).collect();
                let mut scratch = LinearScratch::default();
                for rows in 0..=21 {
                    let x: Vec<f32> = (0..rows * inputs)
                        .map(|index| if index % 13 == 0 { -0.0 } else { stream.unit() })
                        .collect();
                    let mut expected = vec![0.0; rows * outputs];
                    for row in 0..rows {
                        for output in 0..outputs {
                            expected[row * outputs + output] = weights.dot_row(
                                &x[row * inputs..(row + 1) * inputs],
                                output,
                                inputs,
                            ) + bias[output];
                        }
                    }
                    let mut actual = vec![f32::NAN; rows * outputs];
                    linear_into(
                        &x,
                        &weights,
                        &bias,
                        inputs,
                        outputs,
                        &mut actual,
                        &mut scratch,
                    );
                    assert_eq!(
                        bits(&actual),
                        bits(&expected),
                        "{rows}×{inputs} → {outputs}, quantized {quantized}"
                    );
                }
            }
        }
    }
}

#[test]
fn batched_normalizers_match_one_projection_per_state() {
    for model in models() {
        let width = model.config.n_embd;
        let mut stream = Stream(11);
        for rows in 0..=21 {
            let hidden: Vec<f32> = (0..rows * width).map(|_| stream.unit() * 3.0).collect();
            let mut actual = Vec::new();
            model.log_normalizers_into(&hidden, &mut actual);
            let expected: Vec<f32> = hidden
                .chunks_exact(width)
                .map(|row| model.project(row).1)
                .collect();
            assert_eq!(bits(&actual), bits(&expected), "{rows} rows");
        }
    }
}

#[test]
fn batched_score_matches_scoring_each_candidate_alone() {
    for model in models() {
        let mut stream = Stream(23);
        for round in 0..60 {
            let stems = stems(&mut stream);
            let context: String = stems[0].chars().take(stream.below(30)).collect();
            let mut texts: Vec<String> = (0..stream.below(10))
                .map(|_| candidate(&mut stream, &stems))
                .collect();
            if round % 7 == 0 {
                // Longer than the position table, so both the context and the candidate are cut.
                texts.push(stems[1].repeat(4));
            }
            let candidates: Vec<&str> = texts.iter().map(String::as_str).collect();
            let longest = candidates
                .iter()
                .map(|text| text.chars().count())
                .max()
                .unwrap_or(0);
            let prefix = model.run_prefix(&model.prefix_tokens(&context, longest));
            let expected = if candidates.is_empty() {
                Vec::new()
            } else {
                model.score_against(&prefix, &candidates)
            };
            assert_eq!(
                bits(&model.score(&context, &candidates)),
                bits(&expected),
                "round {round}: {context:?} {candidates:?}"
            );
        }
    }
}

/// The decision loop as it was before decisions were planned and batched, verbatim apart from the
/// name: each candidate continues the longest run it shares a prefix with, one at a time.
impl Reranker {
    fn sequential_log_probabilities(&mut self, context: &str, candidates: &[&str]) -> Vec<f32> {
        let longest = candidates
            .iter()
            .map(|text| text.chars().count())
            .max()
            .unwrap_or(0);
        let tokens = self.model.prefix_tokens(context, longest);
        let matched = self
            .cached
            .as_ref()
            .is_some_and(|(cached, _)| *cached == tokens);
        if !matched {
            let prefix = self.model.run_prefix(&tokens);
            self.cached = Some((tokens, prefix));
        }
        let (_, prefix) = self.cached.as_ref().expect("prefix was just computed");
        let mut previous =
            std::mem::replace(&mut self.resume, std::mem::take(&mut self.resume_scratch));
        self.resume.clear();
        if !matched {
            previous.clear();
        }
        let mut scores = Vec::with_capacity(candidates.len());
        let limit = self
            .model
            .config
            .context
            .saturating_sub(prefix.length)
            .max(1);
        for text in candidates {
            let tail = self.model.encode_limited(text, limit);
            if tail.is_empty() {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            if let Some((_, run)) = previous
                .iter()
                .chain(&self.resume)
                .find(|(tokens, _)| tokens == &tail)
            {
                let total = run.cumulative[tail.len() - 1];
                scores.push(total / tail.len() as f32);
                self.resume.push((tail, Arc::clone(run)));
                continue;
            }
            let shared = self
                .resume
                .iter()
                .chain(&previous)
                .map(|(tokens, run)| {
                    let agreed = tokens
                        .iter()
                        .zip(&tail)
                        .take_while(|(left, right)| left == right)
                        .count();
                    (run, agreed.min(tail.len() - 1))
                })
                .max_by_key(|(_, agreed)| *agreed);
            let run = match shared {
                Some((previous, agreed)) if agreed > 0 => {
                    let (from, carried) = previous.resume(agreed, &self.model);
                    let rest = self.model.run_tail(&from, &tail[agreed..]);
                    previous.joined(agreed, carried, rest)
                }
                _ => self.model.run_tail(prefix, &tail),
            };
            let total = run.cumulative[run.cumulative.len() - 1];
            scores.push(total / tail.len() as f32);
            self.resume.push((tail, Arc::new(run)));
        }
        previous.clear();
        self.resume_scratch = previous;
        scores
    }
}

fn same_run(actual: &TailRun, expected: &TailRun) {
    assert_eq!(actual.base, expected.base);
    assert_eq!(actual.pending, expected.pending);
    assert_eq!(bits(&actual.cumulative), bits(&expected.cumulative));
    for (left, right) in actual
        .keys
        .iter()
        .chain(&actual.values)
        .zip(expected.keys.iter().chain(&expected.values))
    {
        assert_eq!(bits(left), bits(right));
        assert_eq!(
            left.capacity(),
            right.capacity(),
            "retained KV capacity changed"
        );
    }
    assert_eq!(actual.hidden.len(), expected.hidden.len());
    for (left, right) in actual.hidden.iter().zip(&expected.hidden) {
        assert_eq!(bits(&left.values), bits(&right.values));
        assert_eq!(
            left.log_normalizer.get().map(|value| value.to_bits()),
            right.log_normalizer.get().map(|value| value.to_bits())
        );
    }
    match (actual.terminal.get(), expected.terminal.get()) {
        (None, None) => {}
        (Some(left), Some(right)) => {
            assert_eq!(bits(&left.hidden.values), bits(&right.hidden.values));
            for (left, right) in left
                .keys
                .iter()
                .chain(&left.values)
                .zip(right.keys.iter().chain(&right.values))
            {
                assert_eq!(bits(left), bits(right));
            }
        }
        (left, right) => panic!(
            "terminal state differs: batched {}, sequential {}",
            left.is_some(),
            right.is_some()
        ),
    }
}

/// Drives both loops through the same decisions and compares everything a later decision can
/// read: the scores, every retained run, and how many positions and projections each one ran.
fn compare_decisions(model: &Arc<SentenceModel>, decisions: &[(String, Vec<String>)]) {
    let mut batched = Reranker::new(model.clone());
    let mut sequential = Reranker::new(model.clone());
    let counts = || {
        (
            FORWARD_POSITIONS.with(std::cell::Cell::get),
            VOCABULARY_PROJECTIONS.with(std::cell::Cell::get),
        )
    };
    for (round, (context, texts)) in decisions.iter().enumerate() {
        let candidates: Vec<&str> = texts.iter().map(String::as_str).collect();
        let before = counts();
        let expected = sequential.sequential_log_probabilities(context, &candidates);
        let middle = counts();
        let actual = batched.log_probabilities(context, &candidates);
        let after = counts();
        assert_eq!(
            bits(&actual),
            bits(&expected),
            "round {round}: {context:?} {candidates:?}"
        );
        assert_eq!(
            (after.0 - middle.0, after.1 - middle.1),
            (middle.0 - before.0, middle.1 - before.1),
            "round {round}: forward positions and projections differ"
        );
        assert_eq!(batched.resume.len(), sequential.resume.len());
        for ((tokens, run), (prior, other)) in batched.resume.iter().zip(&sequential.resume) {
            assert_eq!(tokens, prior);
            same_run(run, other);
        }
        // Runs shared between candidates stay shared, so a later continuation finds one terminal.
        for left in 0..batched.resume.len() {
            for right in left + 1..batched.resume.len() {
                assert_eq!(
                    Arc::ptr_eq(&batched.resume[left].1, &batched.resume[right].1),
                    Arc::ptr_eq(&sequential.resume[left].1, &sequential.resume[right].1)
                );
            }
        }
    }
}

#[test]
fn batched_decisions_match_one_candidate_at_a_time() {
    for model in models() {
        let mut stream = Stream(41);
        let mut decisions = Vec::new();
        let mut context = String::new();
        for _ in 0..12 {
            let stems = stems(&mut stream);
            // Typing: each decision's candidates are the stems a character further on, with the
            // last characters rewritten now and then, as a decoder does.
            for typed in 1..=10 {
                let texts: Vec<String> = (0..stream.below(9) + 1)
                    .map(|_| {
                        let mut text: String = stems[stream.below(stems.len())]
                            .chars()
                            .take(typed)
                            .collect();
                        if stream.below(3) == 0 {
                            text.pop();
                            text.push(ALPHABET[stream.below(ALPHABET.len())]);
                        }
                        if stream.below(5) == 0 {
                            text.push(ALPHABET[stream.below(ALPHABET.len())]);
                        }
                        text
                    })
                    .collect();
                decisions.push((context.clone(), texts));
                if stream.below(10) == 0 {
                    decisions.push((
                        context.clone(),
                        (0..stream.below(5))
                            .map(|_| candidate(&mut stream, &stems))
                            .chain([String::new()])
                            .collect(),
                    ));
                }
            }
            context.push_str(&stems[0][..stems[0].char_indices().nth(3).unwrap().0]);
        }
        compare_decisions(&model, &decisions);
    }
}

#[test]
fn a_batch_reuses_terminals_across_and_within_decisions() {
    let model = model(16, 2, 2, true);
    let decisions: Vec<(String, Vec<String>)> = [
        ("我", vec!["你好", "你好", "你"]),
        // Past the last token of a previous run, twice, and past one of this decision's own runs.
        ("我", vec!["你好吗", "你好很", "你好吗我", "你吗", "你吗好"]),
        // A single character continued by a longer candidate in the same decision.
        ("我", vec!["很", "很好", "很好吗", "很好"]),
        ("", vec!["", "我"]),
        ("我", vec!["你好吗我很", "你好吗我很好", "你"]),
    ]
    .into_iter()
    .map(|(context, texts)| {
        (
            context.to_owned(),
            texts.into_iter().map(str::to_owned).collect(),
        )
    })
    .collect();
    compare_decisions(&model, &decisions);
}

/// The same comparisons on the published weights, over the comparison pool typed one character at
/// a time. Needs `MSIME_NEURAL_MODEL_DIR`; checks `sentence-model-desktop.safetensors` as well when
/// it is there.
#[test]
#[ignore = "需要显式指定固定真实模型"]
fn real_models_batched_decisions_match_one_candidate_at_a_time() {
    let directory = std::env::var_os("MSIME_NEURAL_MODEL_DIR").expect("需要固定模型目录");
    let pool = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../eval/open-models/pool.jsonl"),
    )
    .expect("the comparison pool sits beside the crate in the repository");
    let cases: Vec<Vec<String>> = pool
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let case: serde_json::Value = serde_json::from_str(line).unwrap();
            case["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .map(|text| text.as_str().unwrap().to_owned())
                .collect()
        })
        .collect();
    for name in [
        "sentence-model.safetensors",
        "sentence-model-desktop.safetensors",
    ] {
        let path = std::path::Path::new(&directory).join(name);
        let Ok(bytes) = std::fs::read(&path) else {
            assert_ne!(name, "sentence-model.safetensors", "missing {path:?}");
            continue;
        };
        let model = Arc::new(SentenceModel::load(&bytes).unwrap());
        let mut decisions = Vec::new();
        for (index, candidates) in cases.iter().enumerate() {
            let context = if index % 3 == 0 { "" } else { "今天下午" };
            let longest = candidates
                .iter()
                .map(|text| text.chars().count())
                .max()
                .unwrap();
            for typed in 1..=longest {
                let texts: Vec<String> = candidates
                    .iter()
                    .map(|text| text.chars().take(typed).collect())
                    .collect();
                decisions.push((context.to_owned(), texts));
            }
            let texts: Vec<&str> = candidates.iter().map(String::as_str).collect();
            let prefix = model.run_prefix(&model.prefix_tokens(context, longest));
            assert_eq!(
                bits(&model.score(context, &texts)),
                bits(&model.score_against(&prefix, &texts)),
                "{name}: {texts:?}"
            );
        }
        compare_decisions(&model, &decisions);
    }
}
