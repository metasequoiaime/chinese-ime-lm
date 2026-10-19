//! Reranks candidates the pinyin decoder assembled, using a character-level language model.
//!
//! The model is produced by `neural/`, and the measurements that shaped this crate are recorded in
//! `docs/measurements.md`. Two of them are load-bearing and are enforced here rather than left to callers:
//!
//! Reranking applies only when the engine's leading candidate was assembled by the lattice or the
//! fallback decoder. An exact dictionary hit on the whole key carries corpus frequency the model
//! does not have, and overriding those costs 14 points of top-1 accuracy at every margin tried.
//!
//! Only candidates covering the whole key are comparable. The engine also returns prefixes, and a
//! summed log-probability is larger for fewer characters, so a mixed-length list would rank the
//! shortest candidate first every time regardless of quality. Scores are per character for the same
//! reason, and the caller restricts the set it passes in.

mod weights;

#[cfg(test)]
thread_local! {
    // 只统计当前测试线程实际执行的查询位置和词表投影，不改变生产接口。
    static ATTENTION_POSITIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static VOCABULARY_PROJECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use weights::{Tensor, Weights};

/// A weight matrix, kept in whatever precision the file stored it in.
///
/// Quantized matrices are not expanded on load. A per-output-row scale is a constant factor for the
/// whole row, so it lifts out of the dot product and multiplies the result once — identical
/// arithmetic, a quarter of the memory. The model runs inside an iOS keyboard extension, where
/// turning a 4.5 MB file into 24.5 MB resident is most of what the process is allowed.
struct Matrix {
    inner: Tensor,
}

impl Matrix {
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// `dot(x, row) * scale[row]`, where the scale is 1 for an unquantized matrix.
    fn dot_row(&self, x: &[f32], row: usize, width: usize) -> f32 {
        let span = row * width..(row + 1) * width;
        match &self.inner {
            Tensor::Float(values) => dot(x, &values[span]),
            Tensor::Quantized { values, scales } => dot_i8(x, &values[span]) * scales[row],
        }
    }

    /// 将 token 与位置嵌入直接写入目标行；量化值先乘行缩放，再加位置。
    fn embed_row(&self, index: usize, position: &[f32], out: &mut [f32]) {
        let width = out.len();
        let span = index * width..(index + 1) * width;
        match &self.inner {
            Tensor::Float(values) => {
                for ((out, &token), &position) in out.iter_mut().zip(&values[span]).zip(position) {
                    *out = token + position;
                }
            }
            Tensor::Quantized { values, scales } => {
                let scale = scales[index];
                for ((out, &token), &position) in out.iter_mut().zip(&values[span]).zip(position) {
                    *out = f32::from(token) * scale + position;
                }
            }
        }
    }
}

/// `CandidateSource` values meaning an exact full-key hit in a dictionary. The ordering is defined
/// by `quanpin/word_lattice.h` in the engine: `Database` is 0 and `UserDatabase` is 1.
pub const DICTIONARY_SOURCES: [u8; 2] = [0, 1];

/// What the caller knows about one candidate. Neither field can be worked out from the candidate
/// text, which is why they are asked for rather than derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateFacts {
    /// This candidate answers the whole key, rather than a prefix of it or a predictive completion
    /// running past it. Two candidates that both answer the key are alternatives to each other and
    /// are comparable **even when they are different numbers of characters**: `xian` reads as 现 or
    /// as 西安, and both consume the key. A prefix and a completion answer different keys, so
    /// promoting one over the other would not be a choice between readings.
    ///
    /// Character count cannot answer this. It happens to agree whenever a key has one segmentation,
    /// which is why it served as a proxy, and it disagrees exactly where correction matters.
    ///
    /// An engine that advances composition has already computed this, whether or not it calls it
    /// that: "does this candidate consume the whole key" is the same question as "after selecting it,
    /// is there more input left to answer". Do not write a second test for it — see
    /// `docs/measurements.md` for the mapping onto one engine's existing predicate, and for why
    /// deriving it from the candidate source does not work.
    pub answers_key: bool,
    /// This candidate is an exact dictionary hit **that the engine vouches for as evidence of what
    /// the user meant**, not merely a row that came out of a dictionary.
    ///
    /// Overriding such a hit loses accuracy — 0.719 against 0.690 over 2052 cases, at every margin
    /// tried — and the reason is that a dictionary answering the whole key carries corpus frequency
    /// the model does not have. That reason has a premise: the key it answered is the key the user
    /// meant to type.
    ///
    /// **An engine that corrects input breaks that premise and has to say so here.** When the user
    /// types `gongsi` meaning `gongshi`, 公司 is a perfectly exact hit on the characters that arrived,
    /// and its frequency is evidence for a word nobody asked for. The engine knows it expanded that
    /// key; the crate cannot. So this is not "did a dictionary match" — it is "should the model defer
    /// to this", and only the caller can answer it. An engine offering corrections of a key should
    /// report `false` for hits on the uncorrected reading of that same key.
    pub trusted_dictionary_hit: bool,
}

/// How many comparable candidates are scored. The engine rarely returns more than a handful that
/// answer the same key — two in almost every real case — so this is a ceiling on the worst case
/// rather than a limit that normally binds.
const COMPARED: usize = 9;

/// The bounded set of candidate indices that answer the current key. Keeping the ceiling in the
/// type avoids allocating a temporary `Vec` before every reranking call.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Compared {
    indices: [usize; COMPARED],
    len: usize,
}

impl Compared {
    fn new() -> Self {
        Self {
            indices: [0; COMPARED],
            len: 0,
        }
    }

    fn push(&mut self, index: usize) {
        self.indices[self.len] = index;
        self.len += 1;
    }

    fn as_slice(&self) -> &[usize] {
        &self.indices[..self.len]
    }
}

const UNK: u32 = 1;
const BOS: u32 = 2;

#[cfg(test)]
std::thread_local! {
    static FORWARD_POSITIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // 记录最近一次候选编码的缓冲地址，验证评分时直接转移所有权。
    static LAST_ENCODED_POINTER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LAST_ENCODED_LENGTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model file is truncated")]
    Truncated,
    #[error("model header: {0}")]
    Header(String),
    #[error("model is missing tensor {0}")]
    MissingTensor(String),
    #[error("model is missing metadata {0}")]
    MissingMetadata(String),
    #[error("model geometry: {0}")]
    Geometry(String),
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
struct Config {
    vocab: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    context: usize,
}

struct Block {
    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    qkv_weight: Matrix,
    qkv_bias: Vec<f32>,
    proj_weight: Matrix,
    proj_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
    fc_weight: Matrix,
    fc_bias: Vec<f32>,
    out_weight: Matrix,
    out_bias: Vec<f32>,
}

#[derive(Default)]
struct FeedForwardScratch {
    normalized: Vec<f32>,
    hidden: Vec<f32>,
}

pub struct SentenceModel {
    config: Config,
    index: HashMap<char, u32>,
    token: Matrix,
    position: Vec<f32>,
    blocks: Vec<Block>,
    final_weight: Vec<f32>,
    final_bias: Vec<f32>,
    attribution: String,
    license: String,
}

impl std::fmt::Debug for SentenceModel {
    /// The shape and provenance, never the weights: they run to tens of millions of values and
    /// nothing useful is learned from seeing them.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SentenceModel")
            .field("vocab", &self.config.vocab)
            .field("n_layer", &self.config.n_layer)
            .field("n_head", &self.config.n_head)
            .field("n_embd", &self.config.n_embd)
            .field("context", &self.config.context)
            .field("attribution", &self.attribution)
            .finish()
    }
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Reranker")
            .field("model", &self.model)
            .field("prefix_cached", &self.cached.is_some())
            .finish()
    }
}

// 状态只属于生成它的模型；分叉续跑共享归一化常数，不缓存整张词表。
struct HiddenState {
    values: Vec<f32>,
    log_normalizer: OnceLock<f32>,
}

impl HiddenState {
    fn new(values: Vec<f32>) -> Self {
        Self {
            values,
            log_normalizer: OnceLock::new(),
        }
    }
}

/// 上文持有各层 KV；普通候选续跑借用已有 KV，可见位置由 length 限定。
struct Prefix<'a> {
    length: usize,
    keys: Cow<'a, [Vec<f32>]>,
    values: Cow<'a, [Vec<f32>]>,
    /// 末位置续跑借用单行，直接在最终 KV 中拼接。
    terminal: Option<&'a TerminalState>,
    /// 前缀末位置预测候选首字，隐藏状态与它的归一化常数一起共享。
    last_hidden: Arc<HiddenState>,
}

// 一段已完成前向的位置；累计分仍由调用者按原顺序计算。
struct TailStates {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    hidden: Vec<Arc<HiddenState>>,
}

// 末位置只保存新增的一行 KV，不重复持有整段前缀。
struct TerminalState {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    hidden: Arc<HiddenState>,
}

/// Everything one candidate's run passed through, so that a later run agreeing with part of it can
/// start from where they stop agreeing instead of from the beginning.
struct TailRun {
    /// 上文占用的绝对位置数；KV 仅包含上文及已前向的位置，比累计分少一个候选位置。
    base: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// 已计算位置预测下一字的隐藏状态；末 token 只在续跑经过它时前向。
    hidden: Vec<Arc<HiddenState>>,
    /// 前 i+1 个 token 的完整累计对数概率，保留原来的累加分组。
    cumulative: Vec<f32>,
    pending: u32,
    terminal: OnceLock<TerminalState>,
}

impl TailRun {
    /// 恢复前 taken 个 token 后的前缀；已有 KV 和按需补算的末行都只借用。
    fn resume(&self, taken: usize, model: &SentenceModel) -> (Prefix<'_>, f32) {
        let n_embd = model.config.n_embd;
        debug_assert!(taken > 0 && taken <= self.cumulative.len());
        if taken > self.hidden.len() {
            let terminal = self.terminal.get_or_init(|| {
                let mut states = model.run_tail_states(
                    self.base + self.hidden.len(),
                    &self.keys,
                    &self.values,
                    None,
                    &[self.pending],
                );
                let width = (self.base + self.hidden.len()) * n_embd;
                TerminalState {
                    keys: states
                        .keys
                        .iter_mut()
                        .map(|row| row.split_off(width))
                        .collect(),
                    values: states
                        .values
                        .iter_mut()
                        .map(|row| row.split_off(width))
                        .collect(),
                    hidden: states.hidden.pop().expect("末位置已有前向结果"),
                }
            });
            return (
                Prefix {
                    length: self.base + taken,
                    keys: Cow::Borrowed(&self.keys),
                    values: Cow::Borrowed(&self.values),
                    terminal: Some(terminal),
                    last_hidden: terminal.hidden.clone(),
                },
                self.cumulative[taken - 1],
            );
        }
        (
            Prefix {
                length: self.base + taken,
                keys: Cow::Borrowed(&self.keys),
                values: Cow::Borrowed(&self.values),
                terminal: None,
                last_hidden: self.hidden[taken - 1].clone(),
            },
            self.cumulative[taken - 1],
        )
    }

    /// 把已有前 taken 个位置与 rest 拼为完整候选，供下次按键从任意共享位置续跑。
    fn joined(&self, taken: usize, carried: f32, rest: TailRun) -> TailRun {
        // resume 已补齐经过的末位置，joined 共享它，保留同一份归一化缓存。
        let terminal = if taken > self.hidden.len() {
            Some(
                self.terminal
                    .get()
                    .expect("拼接前已恢复末位置")
                    .hidden
                    .clone(),
            )
        } else {
            None
        };
        TailRun {
            base: self.base,
            keys: rest.keys,
            values: rest.values,
            hidden: self
                .hidden
                .iter()
                .take(taken)
                .cloned()
                .chain(terminal)
                .chain(rest.hidden)
                .collect(),
            cumulative: self.cumulative[..taken]
                .iter()
                .copied()
                .chain(rest.cumulative.iter().map(|value| carried + value))
                .collect(),
            pending: rest.pending,
            terminal: rest.terminal,
        }
    }
}

impl SentenceModel {
    pub fn load(bytes: &[u8]) -> Result<Self, ModelError> {
        let mut weights = Weights::parse(bytes)?;
        let config: Config = serde_json::from_str(weights.metadata("config")?)
            .map_err(|error| ModelError::Header(error.to_string()))?;
        let vocabulary: Vec<String> = serde_json::from_str(weights.metadata("vocab")?)
            .map_err(|error| ModelError::Header(error.to_string()))?;
        let attribution = weights
            .metadata("attribution")
            .unwrap_or_default()
            .to_owned();
        let license = weights.metadata("license").unwrap_or_default().to_owned();

        // Every dimension below is multiplied out while checking the tensors, and all of them come
        // from the file. Bounding them here means the later arithmetic cannot overflow, which is
        // simpler to be sure of than checking each multiplication.
        const LARGEST: usize = 1 << 20;
        if config.vocab > LARGEST
            || config.n_layer > LARGEST
            || config.n_embd > LARGEST
            || config.context > LARGEST
        {
            return Err(ModelError::Geometry(
                "a dimension is larger than any real model".into(),
            ));
        }
        // `%` rather than `is_multiple_of`, which needs a far newer compiler than anything else here.
        if config.n_head == 0 || config.n_embd == 0 || config.n_embd % config.n_head != 0 {
            return Err(ModelError::Geometry(format!(
                "{} channels do not divide into {} heads",
                config.n_embd, config.n_head
            )));
        }
        if vocabulary.len() != config.vocab {
            return Err(ModelError::Geometry(format!(
                "vocabulary holds {} tokens but the configuration says {}",
                vocabulary.len(),
                config.vocab
            )));
        }

        let index = vocabulary
            .iter()
            .enumerate()
            .filter_map(|(id, token)| {
                let mut chars = token.chars();
                match (chars.next(), chars.next()) {
                    // Reserved tokens are spelled `<pad>` and are addressed by constant, not by text.
                    (Some(ch), None) => Some((ch, id as u32)),
                    _ => None,
                }
            })
            .collect();

        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let name = |suffix: &str| format!("blocks.{layer}.{suffix}");
            blocks.push(Block {
                ln1_weight: weights.take(&name("ln1.weight"))?,
                ln1_bias: weights.take(&name("ln1.bias"))?,
                qkv_weight: Matrix {
                    inner: weights.take_tensor(&name("qkv.weight"))?,
                },
                qkv_bias: weights.take(&name("qkv.bias"))?,
                proj_weight: Matrix {
                    inner: weights.take_tensor(&name("proj.weight"))?,
                },
                proj_bias: weights.take(&name("proj.bias"))?,
                ln2_weight: weights.take(&name("ln2.weight"))?,
                ln2_bias: weights.take(&name("ln2.bias"))?,
                fc_weight: Matrix {
                    inner: weights.take_tensor(&name("fc.weight"))?,
                },
                fc_bias: weights.take(&name("fc.bias"))?,
                out_weight: Matrix {
                    inner: weights.take_tensor(&name("out.weight"))?,
                },
                out_bias: weights.take(&name("out.bias"))?,
            });
        }

        let model = Self {
            config,
            index,
            token: Matrix {
                inner: weights.take_tensor("tok.weight")?,
            },
            position: weights.take("pos.weight")?,
            blocks,
            final_weight: weights.take("ln_f.weight")?,
            final_bias: weights.take("ln_f.bias")?,
            attribution,
            license,
        };
        model.check_geometry()?;
        Ok(model)
    }

    fn check_geometry(&self) -> Result<(), ModelError> {
        let Config {
            vocab,
            n_embd,
            context,
            ..
        } = self.config;
        let expect = |name: &str, actual: usize, wanted: usize| {
            (actual == wanted).then_some(()).ok_or_else(|| {
                ModelError::Geometry(format!("{name} holds {actual}, wanted {wanted}"))
            })
        };
        let expect_matrix = |name: &str, matrix: &Matrix, rows: usize, columns: usize| {
            expect(name, matrix.len(), rows * columns)?;
            if let Tensor::Quantized { scales, .. } = &matrix.inner {
                expect(&format!("{name}.scale"), scales.len(), rows)?;
            }
            Ok(())
        };
        expect_matrix("tok.weight", &self.token, vocab, n_embd)?;
        expect("pos.weight", self.position.len(), context * n_embd)?;
        expect("ln_f.weight", self.final_weight.len(), n_embd)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            expect_matrix(
                &format!("blocks.{layer}.qkv"),
                &block.qkv_weight,
                3 * n_embd,
                n_embd,
            )?;
            expect_matrix(
                &format!("blocks.{layer}.proj"),
                &block.proj_weight,
                n_embd,
                n_embd,
            )?;
            expect_matrix(
                &format!("blocks.{layer}.fc"),
                &block.fc_weight,
                4 * n_embd,
                n_embd,
            )?;
            expect_matrix(
                &format!("blocks.{layer}.out"),
                &block.out_weight,
                n_embd,
                4 * n_embd,
            )?;
        }
        Ok(())
    }

    /// The corpus attribution carried in the weights file. Both corpora require it to travel with a
    /// model trained on them, which is why it lives in the file rather than beside it.
    pub fn attribution(&self) -> &str {
        &self.attribution
    }

    /// The licence of the **weights**, as an SPDX identifier, which is not the licence of the code
    /// that produced them. Empty when the file does not say — older files do not, and a caller that
    /// needs to know should treat silence as unknown rather than as permission.
    pub fn license(&self) -> &str {
        &self.license
    }

    pub fn context_length(&self) -> usize {
        self.config.context
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_limited(text, usize::MAX)
    }

    /// 模型不会评分窗口之后的 token，编码和保留容量都限制在这个窗口内。
    fn encode_limited(&self, text: &str, limit: usize) -> Vec<u32> {
        let tokens: Vec<u32> = text
            .chars()
            .take(limit)
            .map(|ch| self.index.get(&ch).copied().unwrap_or(UNK))
            .collect();
        #[cfg(test)]
        LAST_ENCODED_POINTER.with(|pointer| pointer.set(tokens.as_ptr() as usize));
        #[cfg(test)]
        LAST_ENCODED_LENGTH.with(|length| length.set(tokens.len()));
        tokens
    }

    /// Mean log-probability per character of each candidate, conditioned on the committed text.
    ///
    /// The mean rather than the sum, because candidates of different lengths are otherwise not
    /// comparable. Callers are expected to pass only candidates that cover the whole key, which
    /// makes the lengths equal in practice; normalizing anyway means a caller that gets that wrong
    /// gets a degraded ranking rather than a meaningless one.
    pub fn score(&self, context: &str, candidates: &[&str]) -> Vec<f32> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let longest = candidates
            .iter()
            .map(|text| text.chars().count())
            .max()
            .unwrap_or(0);
        let prefix = self.run_prefix(&self.prefix_tokens(context, longest));
        self.score_against(&prefix, candidates)
    }

    /// The committed text as tokens, opened with `BOS` and trimmed to leave room for the longest
    /// candidate.
    ///
    /// The position table is the hard limit on sequence length, so the context yields rather than
    /// the candidate: a truncated candidate would be scored on text the user did not type, whereas
    /// a truncated context only weakens the conditioning.
    fn prefix_tokens(&self, context: &str, longest: usize) -> Vec<u32> {
        let room = self.config.context.saturating_sub(longest + 1);
        let encoded = self.encode(context);
        let start = encoded.len().saturating_sub(room);
        let mut tokens = Vec::with_capacity(encoded.len() - start + 1);
        tokens.push(BOS);
        tokens.extend_from_slice(&encoded[start..]);
        tokens
    }

    fn score_against(&self, prefix: &Prefix<'_>, candidates: &[&str]) -> Vec<f32> {
        candidates
            .iter()
            .map(|text| {
                let tail = self.encode(text);
                if tail.is_empty() {
                    return f32::NEG_INFINITY;
                }
                let limit = self.config.context.saturating_sub(prefix.length).max(1);
                let tail = &tail[..tail.len().min(limit)];
                self.score_tail(prefix, tail) / tail.len() as f32
            })
            .collect()
    }

    fn embed(&self, tokens: &[u32], offset: usize) -> Vec<f32> {
        #[cfg(test)]
        FORWARD_POSITIONS.with(|count| count.set(count.get() + tokens.len()));
        let n_embd = self.config.n_embd;
        let mut x = vec![0.0; tokens.len() * n_embd];
        for (step, &token) in tokens.iter().enumerate() {
            let token = (token as usize).min(self.config.vocab - 1);
            let position = (offset + step).min(self.config.context - 1);
            let row = &mut x[step * n_embd..(step + 1) * n_embd];
            let position_row = &self.position[position * n_embd..(position + 1) * n_embd];
            self.token.embed_row(token, position_row, row);
        }
        x
    }

    /// Runs the prefix and keeps every layer's keys and values.
    fn run_prefix(&self, tokens: &[u32]) -> Prefix<'static> {
        let n_embd = self.config.n_embd;
        let mut x = self.embed(tokens, 0);
        let mut feed_forward = FeedForwardScratch::default();
        let mut keys = Vec::with_capacity(self.blocks.len());
        let mut values = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            self.normalize_attention_into(&x, block, &mut feed_forward);
            self.qkv_into(block, &mut feed_forward);
            let (k, v) = split_qkv_with_suffix_into(
                &feed_forward.hidden,
                tokens.len(),
                n_embd,
                &[],
                &[],
                None,
                &mut feed_forward.normalized,
            );
            // 最后一层只消费末位置输出；所有位置的 KV 仍供后续候选使用。
            let first = if layer + 1 == self.blocks.len() {
                tokens.len() - 1
            } else {
                0
            };
            let start = first * n_embd;
            let attended = self.attention(
                &feed_forward.normalized[start..],
                &k,
                &v,
                tokens.len() - first,
                first,
            );
            self.project_attention_into(&attended, block, &mut feed_forward);
            for (slot, &value) in x[start..].iter_mut().zip(&feed_forward.normalized) {
                *slot += value;
            }
            self.feed_forward(&mut x[start..], block, &mut feed_forward);
            keys.push(k);
            values.push(v);
        }
        let last = (tokens.len() - 1) * n_embd;
        feed_forward.normalized.resize(n_embd, 0.0);
        layer_norm_into(
            &x[last..],
            &self.final_weight,
            &self.final_bias,
            n_embd,
            &mut feed_forward.normalized,
        );
        let last_hidden = Arc::new(HiddenState::new(feed_forward.normalized));
        Prefix {
            length: tokens.len(),
            keys: Cow::Owned(keys),
            values: Cow::Owned(values),
            terminal: None,
            last_hidden,
        }
    }

    /// Runs a candidate against a cached prefix and returns its total log-probability.
    fn score_tail(&self, prefix: &Prefix<'_>, tail: &[u32]) -> f32 {
        let run = self.run_tail(prefix, tail);
        run.cumulative[run.cumulative.len() - 1]
    }

    /// 完整累计分只需要前 length-1 个隐藏状态；末位置留给真正经过它的续跑。
    ///
    /// 分段累计仍沿用既有分组：carried + rest，不改变 token 的先后顺序。
    fn run_tail(&self, prefix: &Prefix<'_>, tail: &[u32]) -> TailRun {
        let states = self.run_tail_states(
            prefix.length,
            &prefix.keys,
            &prefix.values,
            prefix.terminal,
            &tail[..tail.len() - 1],
        );
        let mut running = self.log_probability(&prefix.last_hidden, tail[0]);
        let mut cumulative = Vec::with_capacity(tail.len());
        cumulative.push(running);
        for (state, &next) in states.hidden.iter().zip(&tail[1..]) {
            running += self.log_probability(state, next);
            cumulative.push(running);
        }
        TailRun {
            base: prefix.length,
            keys: states.keys,
            values: states.values,
            hidden: states.hidden,
            cumulative,
            pending: *tail.last().expect("候选至少有一个 token"),
            terminal: OnceLock::new(),
        }
    }

    /// 仅计算需要的因果位置；空段只保留上文 KV，单字评分无需前向。
    fn run_tail_states(
        &self,
        length: usize,
        prefix_keys: &[Vec<f32>],
        prefix_values: &[Vec<f32>],
        terminal: Option<&TerminalState>,
        tokens: &[u32],
    ) -> TailStates {
        // 借用的旧 KV 可能更长，只复制本次续跑可见的位置。
        let n_embd = self.config.n_embd;
        let prefix_width = (length - usize::from(terminal.is_some())) * n_embd;
        if tokens.is_empty() {
            let copy = |rows: &[Vec<f32>], last: Option<&[Vec<f32>]>| {
                rows.iter()
                    .enumerate()
                    .map(|(layer, row)| {
                        let mut out = Vec::with_capacity(length * n_embd);
                        out.extend_from_slice(&row[..prefix_width]);
                        if let Some(last) = last {
                            out.extend_from_slice(&last[layer]);
                        }
                        out
                    })
                    .collect()
            };
            return TailStates {
                keys: copy(prefix_keys, terminal.map(|last| last.keys.as_slice())),
                values: copy(prefix_values, terminal.map(|last| last.values.as_slice())),
                hidden: Vec::new(),
            };
        }
        let mut x = self.embed(tokens, length);
        let mut feed_forward = FeedForwardScratch::default();
        let mut keys_out = Vec::with_capacity(self.blocks.len());
        let mut values_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            self.normalize_attention_into(&x, block, &mut feed_forward);
            self.qkv_into(block, &mut feed_forward);
            let (keys, values) = split_qkv_with_suffix_into(
                &feed_forward.hidden,
                tokens.len(),
                n_embd,
                &prefix_keys[layer][..prefix_width],
                &prefix_values[layer][..prefix_width],
                terminal.map(|last| (last.keys[layer].as_slice(), last.values[layer].as_slice())),
                &mut feed_forward.normalized,
            );
            let attended = self.attention(
                &feed_forward.normalized,
                &keys,
                &values,
                tokens.len(),
                length,
            );
            self.project_attention_into(&attended, block, &mut feed_forward);
            for (slot, &value) in x.iter_mut().zip(&feed_forward.normalized) {
                *slot += value;
            }
            self.feed_forward(&mut x, block, &mut feed_forward);
            keys_out.push(keys);
            values_out.push(values);
        }
        feed_forward.normalized.resize(x.len(), 0.0);
        layer_norm_into(
            &x,
            &self.final_weight,
            &self.final_bias,
            n_embd,
            &mut feed_forward.normalized,
        );

        TailStates {
            keys: keys_out,
            values: values_out,
            hidden: feed_forward
                .normalized
                .chunks_exact(n_embd)
                .map(|row| Arc::new(HiddenState::new(row.to_vec())))
                .collect(),
        }
    }

    fn project_attention_into(
        &self,
        attended: &[f32],
        block: &Block,
        scratch: &mut FeedForwardScratch,
    ) {
        scratch.normalized.resize(attended.len(), 0.0);
        linear_into(
            attended,
            &block.proj_weight,
            &block.proj_bias,
            self.config.n_embd,
            self.config.n_embd,
            &mut scratch.normalized,
        );
    }

    fn normalize_attention_into(&self, x: &[f32], block: &Block, scratch: &mut FeedForwardScratch) {
        scratch.normalized.resize(x.len(), 0.0);
        layer_norm_into(
            x,
            &block.ln1_weight,
            &block.ln1_bias,
            self.config.n_embd,
            &mut scratch.normalized,
        );
    }

    fn qkv_into(&self, block: &Block, scratch: &mut FeedForwardScratch) {
        let normalized = &scratch.normalized;
        scratch.hidden.resize(normalized.len() * 3, 0.0);
        linear_into(
            normalized,
            &block.qkv_weight,
            &block.qkv_bias,
            self.config.n_embd,
            3 * self.config.n_embd,
            &mut scratch.hidden,
        );
    }

    fn feed_forward(&self, x: &mut [f32], block: &Block, scratch: &mut FeedForwardScratch) {
        let n_embd = self.config.n_embd;
        scratch.normalized.resize(x.len(), 0.0);
        layer_norm_into(
            x,
            &block.ln2_weight,
            &block.ln2_bias,
            n_embd,
            &mut scratch.normalized,
        );
        scratch.hidden.resize(x.len() * 4, 0.0);
        linear_into(
            &scratch.normalized,
            &block.fc_weight,
            &block.fc_bias,
            n_embd,
            4 * n_embd,
            &mut scratch.hidden,
        );
        for value in &mut scratch.hidden {
            *value = gelu(*value);
        }
        // 第一线性层已消费归一化结果，输出投影可以覆盖同一份存储。
        linear_into(
            &scratch.hidden,
            &block.out_weight,
            &block.out_bias,
            4 * n_embd,
            n_embd,
            &mut scratch.normalized,
        );
        for (slot, &value) in x.iter_mut().zip(&scratch.normalized) {
            *slot += value;
        }
    }

    /// 同一隐藏状态的完整词表只投影一次，后续 token 只计算自己的 logit。
    fn log_probability(&self, hidden: &HiddenState, token: u32) -> f32 {
        let token = (token as usize).min(self.config.vocab - 1);
        let offset = hidden
            .log_normalizer
            .get_or_init(|| self.project(&hidden.values).1);
        self.token
            .dot_row(&hidden.values, token, self.config.n_embd)
            - offset
    }

    /// 全词表 logits 与归一化常数保持既有的点积、最高值和指数求和顺序。
    fn project(&self, hidden: &[f32]) -> (Vec<f32>, f32) {
        #[cfg(test)]
        VOCABULARY_PROJECTIONS.with(|count| count.set(count.get() + 1));
        let n_embd = self.config.n_embd;
        let mut logits = Vec::with_capacity(self.config.vocab);
        let mut highest = f32::NEG_INFINITY;
        match &self.token.inner {
            Tensor::Float(values) => {
                for row in values.chunks_exact(n_embd).take(self.config.vocab) {
                    let logit = dot(hidden, row);
                    highest = highest.max(logit);
                    logits.push(logit);
                }
            }
            Tensor::Quantized { values, scales } => {
                for (row, &scale) in values
                    .chunks_exact(n_embd)
                    .zip(&scales[..self.config.vocab])
                    .take(self.config.vocab)
                {
                    let logit = dot_i8(hidden, row) * scale;
                    highest = highest.max(logit);
                    logits.push(logit);
                }
            }
        }
        let sum: f32 = logits.iter().map(|logit| (logit - highest).exp()).sum();
        let offset = highest + sum.ln();
        (logits, offset)
    }

    /// 返回每个 token 的对数概率，公开的完整分布入口仍计算所有行。
    fn distribution(&self, hidden: &[f32]) -> Vec<f32> {
        let (mut logits, offset) = self.project(hidden);
        for logit in &mut logits {
            *logit -= offset;
        }
        logits
    }

    /// log P(next character | text), over the whole vocabulary.
    ///
    /// This is the primitive a search needs: reranking asks the model to judge finished strings,
    /// while a search asks it what should come next, which is what lets it reach a reading the
    /// engine never proposed.
    pub fn next_log_probabilities(&self, text: &str) -> Vec<f32> {
        let mut tokens = vec![BOS];
        let encoded = self.encode(text);
        let room = self.config.context - 1;
        tokens.extend_from_slice(&encoded[encoded.len().saturating_sub(room)..]);
        let prefix = self.run_prefix(&tokens);
        self.distribution(&prefix.last_hidden.values)
    }

    /// The identifier this model uses for `character`, when it has one.
    pub fn token(&self, character: char) -> Option<u32> {
        self.index.get(&character).copied()
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], rows: usize, offset: usize) -> Vec<f32> {
        #[cfg(test)]
        ATTENTION_POSITIONS.with(|count| count.set(count.get() + rows));
        let n_embd = self.config.n_embd;
        let heads = self.config.n_head;
        let width = n_embd / heads;
        let scale = 1.0 / (width as f32).sqrt();
        let total = k.len() / n_embd;
        let mut out = vec![0.0; rows * n_embd];
        let mut scores = vec![0.0; total];
        for row in 0..rows {
            // Causal: a query at absolute position `offset + row` may attend to itself and earlier.
            let visible = offset + row + 1;
            for head in 0..heads {
                let base = head * width;
                let query = &q[row * n_embd + base..row * n_embd + base + width];
                let mut highest = f32::NEG_INFINITY;
                for (step, score) in scores[..visible].iter_mut().enumerate() {
                    let key = &k[step * n_embd + base..step * n_embd + base + width];
                    *score = dot(query, key) * scale;
                    highest = highest.max(*score);
                }
                let mut sum = 0.0;
                for score in &mut scores[..visible] {
                    *score = (*score - highest).exp();
                    sum += *score;
                }
                let target = &mut out[row * n_embd + base..row * n_embd + base + width];
                for (step, &score) in scores[..visible].iter().enumerate() {
                    let weight = score / sum;
                    let value = &v[step * n_embd + base..step * n_embd + base + width];
                    for (slot, &item) in target.iter_mut().zip(value) {
                        *slot += weight * item;
                    }
                }
            }
        }
        out
    }
}

/// A model plus the prefix it last ran, so that typing does not re-run the committed text.
///
/// The prefix is the expensive half of a short decision — nineteen characters of context cost about
/// as much as the candidates themselves — and it only changes when something is committed, not on
/// every keystroke. Holding it across keystrokes is what puts the decision inside a frame.
pub struct Reranker {
    /// Shared, because the weights are read-only and a host opens a session per focus change.
    /// Only the cached prefix belongs to one session.
    model: Arc<SentenceModel>,
    cached: Option<(Vec<u32>, Prefix<'static>)>,
    /// What the previous decision ended up knowing, one entry per candidate it scored: the tokens
    /// it ran, the state it left the model in, and the log-probability accumulated over them.
    ///
    /// A keystroke rescores candidates that are mostly the previous keystroke's candidates with a
    /// character added, so most of each candidate has already been run. Measured on the sentence
    /// eval driven through the real runtime, 61% of the characters scored are a prefix of
    /// something the previous keystroke scored. Keeping only the previous decision bounds this to
    /// one entry per candidate; keeping more would grow without a ceiling for reuse that was not
    /// measured to be there.
    resume: Vec<(Vec<u32>, Arc<TailRun>)>,
    /// A spare vector whose allocation is reused while the previous decision remains available
    /// for prefix matching.
    resume_scratch: Vec<(Vec<u32>, Arc<TailRun>)>,
    /// `best_where` only needs scores until it has chosen an index; keep that short result buffer
    /// between keystrokes instead of allocating it for every decision.
    scores_scratch: Vec<f32>,
}

impl Reranker {
    pub fn new(model: Arc<SentenceModel>) -> Self {
        Self {
            model,
            cached: None,
            resume: Vec::new(),
            resume_scratch: Vec::new(),
            scores_scratch: Vec::new(),
        }
    }

    pub fn model(&self) -> &SentenceModel {
        &self.model
    }

    /// The index of the candidate the model prefers over the engine's leader, or `None` to leave the
    /// order alone. Both slices are the engine's candidate list in its own order, and the returned
    /// index refers to that list.
    ///
    /// **This infers comparability from character count, which is only a proxy, and it is the wrong
    /// proxy as soon as the engine corrects what was typed.** Candidates of the leader's length are
    /// assumed to have answered the same key. That holds while one key has one segmentation, and
    /// breaks when it does not: `xian` reads as 现 or as 西安, both consuming the whole key, and this
    /// method compares only whichever of the two matches the leader. Fuzzy pinyin also makes the
    /// dictionary test wrong here, because `sources` cannot say whether a dictionary hit landed on
    /// the key the user typed or on a fuzz-expanded variant of it.
    ///
    /// Kept because it is the one call an engine using the same source numbering can make without
    /// supplying anything else. An engine that corrects input should use [`Reranker::best_where`],
    /// which asks for both facts instead of guessing at them.
    pub fn best(&mut self, context: &str, texts: &[&str], sources: &[u8]) -> Option<usize> {
        let width = texts.first().map_or(0, |text| text.chars().count());
        self.best_where(context, texts, |index| CandidateFacts {
            answers_key: texts
                .get(index)
                .is_some_and(|text| text.chars().count() == width),
            trusted_dictionary_hit: sources
                .get(index)
                .is_some_and(|source| DICTIONARY_SOURCES.contains(source)),
        })
    }

    /// The same decision with the caller supplying what only the engine knows about each candidate.
    ///
    /// See [`CandidateFacts`] for what the two fields mean and why neither can be derived from the
    /// candidate strings. The comparable set is every candidate that answers the key, capped at nine;
    /// the decision is declined when the leader is an exact dictionary hit on the typed key, when the
    /// leader does not itself answer the key, or when fewer than two candidates do.
    ///
    /// **Candidates that answer the key are compared even when they are different numbers of
    /// characters.** The scores are per-character means for exactly this reason. A summed
    /// log-probability would rank the shortest candidate first regardless of quality, which is what
    /// makes a mixed-length list dangerous — but "mixed length" was never the thing to exclude.
    /// Prefixes and predictive completions have to be excluded because they answer a different key,
    /// not because of their length, and `answers_key` says so directly.
    pub fn best_where(
        &mut self,
        context: &str,
        texts: &[&str],
        facts: impl Fn(usize) -> CandidateFacts,
    ) -> Option<usize> {
        let considered = comparable(texts.len(), &facts)?;
        let mut scores = std::mem::take(&mut self.scores_scratch);
        self.score_indices_into(
            context,
            texts,
            considered.as_slice().iter().copied(),
            &mut scores,
        );
        let mut best = 0;
        for (index, score) in scores.iter().enumerate() {
            if *score > scores[best] {
                best = index;
            }
        }
        let promoted = (best != 0).then(|| considered.as_slice()[best]);
        self.scores_scratch = scores;
        promoted
    }

    /// Mean log-probability per character for each candidate, in the order given.
    ///
    /// Exposed because the model's opinion is only one term in the decision: combining it with the
    /// evidence the engine already has is a different question from letting it decide alone, and
    /// answering that question needs the numbers rather than the verdict.
    pub fn log_probabilities(&mut self, context: &str, candidates: &[&str]) -> Vec<f32> {
        self.score(context, candidates)
    }

    fn score(&mut self, context: &str, candidates: &[&str]) -> Vec<f32> {
        let mut scores = Vec::new();
        self.score_indices_into(context, candidates, 0..candidates.len(), &mut scores);
        scores
    }

    fn score_indices_into(
        &mut self,
        context: &str,
        candidates: &[&str],
        indices: impl Iterator<Item = usize> + Clone,
        scores: &mut Vec<f32>,
    ) {
        let longest = indices
            .clone()
            .map(|index| candidates[index].chars().count())
            .max()
            .unwrap_or(0);
        // Keyed on the tokens rather than the context string, because two different contexts that
        // truncate to the same window are the same prefix and must not invalidate each other.
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
        let capacity = indices.size_hint().0;
        let mut previous =
            std::mem::replace(&mut self.resume, std::mem::take(&mut self.resume_scratch));
        self.resume.clear();
        // Every resume point hangs off the old prefix and means nothing against a new one.
        if !matched {
            previous.clear();
        }
        scores.clear();
        scores.reserve_exact(capacity);
        let limit = self
            .model
            .config
            .context
            .saturating_sub(prefix.length)
            .max(1);
        for index in indices {
            let tail = self.model.encode_limited(candidates[index], limit);
            if tail.is_empty() {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            // 完全命中优先保留旧轮状态，再复用本轮已有结果；两组状态都供后续候选借用。
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

            // 选两轮的最长共享前缀；max_by_key 取最后一个同长项，旧轮顺序和优先级保持。
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
    }
}

/// Which candidates may be compared, or `None` when the decision should be declined.
///
/// Separate from scoring and free of the model, because this is the part that decides what the
/// measurements mean: every published accuracy number for this crate is a consequence of which
/// candidates were allowed into the comparison, so it is worth being able to test on its own.
fn comparable(count: usize, facts: &impl Fn(usize) -> CandidateFacts) -> Option<Compared> {
    if count == 0 {
        return None;
    }
    let leader = facts(0);
    // An exact hit on the typed key carries frequency evidence the model does not have. A leader
    // that does not answer the key leaves nothing to promote it over: the comparison would be
    // between readings of different keys.
    if leader.trusted_dictionary_hit || !leader.answers_key {
        return None;
    }
    let mut considered = Compared::new();
    for index in (0..count).filter(|&index| facts(index).answers_key) {
        if considered.len == COMPARED {
            break;
        }
        considered.push(index);
    }
    (considered.len >= 2).then_some(considered)
}

/// Whether the model should be consulted at all, given the sources the engine reported in order.
///
/// Only the leading candidate matters. When it is an exact dictionary hit on the whole key it
/// carries corpus frequency the model does not have, and reranking measurably loses accuracy.
///
/// **Sources alone cannot answer this for an engine that corrects input.** A hit on a fuzz-expanded
/// key reports the same source as a hit on the typed key while carrying frequency for a word the
/// user did not ask for. Such an engine should decide with [`CandidateFacts::trusted_dictionary_hit`]
/// instead, which asks the question this function can only approximate.
pub fn should_rerank(sources: &[u8]) -> bool {
    match sources.first() {
        Some(source) => !DICTIONARY_SOURCES.contains(source),
        None => false,
    }
}

/// Independent accumulators rather than one running sum.
///
/// A single accumulator makes every multiply-add wait for the previous one, so the loop runs at the
/// latency of the fused multiply-add instead of its throughput and leaves most of the core idle.
/// Eight partial sums give the pipeline eight independent chains to interleave. The summation order
/// differs from a serial reduction, which is immaterial here: the result feeds a ranking, and the
/// order was never guaranteed to match the trainer's anyway.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut partial = [0.0f32; LANES];
    let left = a.chunks_exact(LANES);
    let right = b.chunks_exact(LANES);
    let (tail_left, tail_right) = (left.remainder(), right.remainder());
    for (x, y) in left.zip(right) {
        for lane in 0..LANES {
            partial[lane] += x[lane] * y[lane];
        }
    }
    let mut total = partial.iter().sum::<f32>();
    for (x, y) in tail_left.iter().zip(tail_right) {
        total += x * y;
    }
    total
}

/// `out[m][o] = dot(x[m], w[o]) + bias[o]`, with both operands laid out row-major so every dot
/// product runs over contiguous memory.
#[cfg(test)]
fn linear(x: &[f32], w: &Matrix, bias: &[f32], inputs: usize, outputs: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len() / inputs * outputs];
    linear_into(x, w, bias, inputs, outputs, &mut out);
    out
}

fn linear_into(
    x: &[f32],
    w: &Matrix,
    bias: &[f32],
    inputs: usize,
    outputs: usize,
    out: &mut [f32],
) {
    let rows = x.len() / inputs;
    for row in 0..rows {
        let source = &x[row * inputs..(row + 1) * inputs];
        let target = &mut out[row * outputs..(row + 1) * outputs];
        for (index, slot) in target.iter_mut().enumerate() {
            *slot = w.dot_row(source, index, inputs) + bias[index];
        }
    }
}

/// The quantized counterpart of `dot`, with the same eight independent accumulators and the same
/// reason for them. The row's scale is applied by the caller, once, rather than to every value:
/// that is the whole point of keeping the weights as `i8`.
fn dot_i8(a: &[f32], b: &[i8]) -> f32 {
    const LANES: usize = 8;
    let mut partial = [0.0f32; LANES];
    let left = a.chunks_exact(LANES);
    let right = b.chunks_exact(LANES);
    let (tail_left, tail_right) = (left.remainder(), right.remainder());
    for (x, y) in left.zip(right) {
        for lane in 0..LANES {
            partial[lane] += x[lane] * f32::from(y[lane]);
        }
    }
    let mut total = partial.iter().sum::<f32>();
    for (x, y) in tail_left.iter().zip(tail_right) {
        total += x * f32::from(*y);
    }
    total
}

#[cfg(test)]
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    layer_norm_into(x, weight, bias, width, &mut out);
    out
}

fn layer_norm_into(x: &[f32], weight: &[f32], bias: &[f32], width: usize, out: &mut [f32]) {
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
}

/// 按最终大小拆分，新增 K、V 直接写入已有可见前缀之后。
#[cfg(test)]
fn split_qkv(
    qkv: &[f32],
    rows: usize,
    n_embd: usize,
    prefix_keys: &[f32],
    prefix_values: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    split_qkv_with_suffix(qkv, rows, n_embd, prefix_keys, prefix_values, None)
}

/// 可选末行与旧前缀一起写入最终 K、V，不生成连续的临时前缀。
#[cfg(test)]
fn split_qkv_with_suffix(
    qkv: &[f32],
    rows: usize,
    n_embd: usize,
    prefix_keys: &[f32],
    prefix_values: &[f32],
    suffix: Option<(&[f32], &[f32])>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut query = Vec::with_capacity(rows * n_embd);
    let (keys, values) = split_qkv_with_suffix_into(
        qkv,
        rows,
        n_embd,
        prefix_keys,
        prefix_values,
        suffix,
        &mut query,
    );
    (query, keys, values)
}

/// 把 query 写入调用者提供的 scratch，并按既有顺序返回最终 K、V。
fn split_qkv_with_suffix_into(
    qkv: &[f32],
    rows: usize,
    n_embd: usize,
    prefix_keys: &[f32],
    prefix_values: &[f32],
    suffix: Option<(&[f32], &[f32])>,
    query: &mut Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let (last_keys, last_values) = suffix.unwrap_or((&[], &[]));
    let width = rows * n_embd;
    query.clear();
    query.reserve(width);
    let mut k = Vec::with_capacity(prefix_keys.len() + last_keys.len() + width);
    let mut v = Vec::with_capacity(prefix_values.len() + last_values.len() + width);
    k.extend_from_slice(prefix_keys);
    v.extend_from_slice(prefix_values);
    k.extend_from_slice(last_keys);
    v.extend_from_slice(last_values);
    for row in 0..rows {
        let source = &qkv[row * 3 * n_embd..(row + 1) * 3 * n_embd];
        query.extend_from_slice(&source[..n_embd]);
        k.extend_from_slice(&source[n_embd..2 * n_embd]);
        v.extend_from_slice(&source[2 * n_embd..]);
    }
    (k, v)
}

/// The exact GELU, matching `torch.nn.functional.gelu` with its default settings. The tanh
/// approximation is a different function and the model was not trained with it.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz and Stegun 7.1.26, whose error stays below 1.5e-7 across the range.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    mod bounded_encoding;
    mod project_rows;

    use super::{comparable, erf, gelu, should_rerank, CandidateFacts, Compared};

    // 一层小模型只使用合成权重，缓存测试不读取外部模型文件。
    fn cache_test_model() -> std::sync::Arc<super::SentenceModel> {
        use super::{Block, Config, Matrix, SentenceModel};
        let matrix = |size| Matrix {
            inner: super::Tensor::Float(
                (0..size).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect(),
            ),
        };
        std::sync::Arc::new(SentenceModel {
            config: Config {
                vocab: 8,
                n_layer: 1,
                n_head: 2,
                n_embd: 4,
                context: 8,
            },
            index: ['你', '好', '吗', '我', '很']
                .into_iter()
                .enumerate()
                .map(|(index, character)| (character, index as u32 + 3))
                .collect(),
            token: matrix(8 * 4),
            position: (0..8 * 4)
                .map(|i| (((i * 7) % 19) as f32 - 9.0) / 64.0)
                .collect(),
            blocks: vec![Block {
                ln1_weight: vec![1.0; 4],
                ln1_bias: vec![0.0; 4],
                qkv_weight: matrix(12 * 4),
                qkv_bias: vec![0.0; 12],
                proj_weight: matrix(4 * 4),
                proj_bias: vec![0.0; 4],
                ln2_weight: vec![1.0; 4],
                ln2_bias: vec![0.0; 4],
                fc_weight: matrix(16 * 4),
                fc_bias: vec![0.0; 16],
                out_weight: matrix(4 * 16),
                out_bias: vec![0.0; 4],
            }],
            final_weight: vec![1.0; 4],
            final_bias: vec![0.0; 4],
            attribution: String::new(),
            license: String::new(),
        })
    }

    #[test]
    fn cold_batch_shares_forward_positions_and_completed_runs() {
        let mut reranker = super::Reranker::new(cache_test_model());
        let positions = || super::FORWARD_POSITIONS.with(std::cell::Cell::get);
        let projections = || super::VOCABULARY_PROJECTIONS.with(std::cell::Cell::get);
        let candidates = ["你好我", "你好很", "你好我", "你好我吗"];
        for context in ["我", "很"] {
            let before = (positions(), projections());
            let scores = reranker.log_probabilities(context, &candidates);
            assert_eq!(positions() - before.0, 5, "同批共享前缀仍重复前向");
            assert_eq!(projections() - before.1, 4, "同批共享状态仍重复投影");
            assert_eq!(scores[0].to_bits(), scores[2].to_bits());
            assert!(std::sync::Arc::ptr_eq(
                &reranker.resume[0].1,
                &reranker.resume[2].1
            ));
            for (_, run) in &reranker.resume[1..] {
                assert!(std::sync::Arc::ptr_eq(
                    &reranker.resume[0].1.hidden[0],
                    &run.hidden[0]
                ));
            }
            let retained = std::sync::Arc::downgrade(&reranker.resume[0].1);
            let before = positions();
            reranker.log_probabilities(context, &["你好我吗", "你好我", "你好很"]);
            assert_eq!(positions() - before, 0, "重排完全命中仍做前向");
            reranker.log_probabilities(context, &[]);
            assert!(retained.upgrade().is_none(), "本轮复用不应延长缓存寿命");
        }
    }

    #[test]
    fn same_length_reuse_keeps_the_previous_round_tie_order() {
        let model = cache_test_model();
        let mut reranker = super::Reranker::new(model.clone());
        reranker.log_probabilities("我", &["你好", "你吗"]);
        // 构造有效但独立的旧轮状态，让地址能区分同长匹配的来源。
        let prefix = &reranker.cached.as_ref().unwrap().1;
        for (tokens, run) in &mut reranker.resume {
            *run = std::sync::Arc::new(model.run_tail(prefix, tokens));
        }
        let expected = reranker.resume[1].1.hidden[0].clone();
        reranker.log_probabilities("我", &["你很", "你我"]);
        for (_, run) in &reranker.resume {
            assert!(std::sync::Arc::ptr_eq(&expected, &run.hidden[0]));
        }
    }

    #[test]
    fn terminal_forward_waits_for_a_shared_continuation() {
        let mut reranker = super::Reranker::new(cache_test_model());
        let count = || super::FORWARD_POSITIONS.with(std::cell::Cell::get);
        let before = count();
        reranker.log_probabilities("我", &["你好", "你吗"]);
        assert_eq!(
            count() - before,
            3,
            "上文和 BOS 加共享首字，候选末字无需前向"
        );

        let before = count();
        reranker.log_probabilities("我", &["你很", "你我"]);
        assert_eq!(count() - before, 0, "只改末字仍做了前向");

        let before = count();
        reranker.log_probabilities("我", &["你很好", "你很吗"]);
        assert_eq!(count() - before, 1, "同一末位置应只按需补算一次");

        let before = count();
        reranker.log_probabilities("我", &["你很好我", "你很好吗"]);
        assert_eq!(count() - before, 1, "拼接后的末位置仍应共享补算");

        let before = count();
        reranker.log_probabilities("我", &["你很好", "你", "你很好"]);
        assert_eq!(count() - before, 0, "缩短候选不应补算末位置");
    }

    #[test]
    fn split_qkv_preserves_prefix_and_row_order() {
        let qkv = [
            1.0, -0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let keys = [21.0, 22.0, 23.0, 24.0];
        let values = [31.0, 32.0, 33.0, 34.0];
        let bits = |row: &[f32]| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        for (prefix_keys, prefix_values) in [(&[][..], &[][..]), (&keys[..], &values[..])] {
            let (q, k, v) = super::split_qkv(&qkv, 2, 2, prefix_keys, prefix_values);
            assert_eq!(bits(&q), bits(&[1.0, -0.0, 7.0, 8.0]));
            let expected_keys = [prefix_keys, &[3.0, 4.0, 9.0, 10.0]].concat();
            let expected_values = [prefix_values, &[5.0, 6.0, 11.0, 12.0]].concat();
            assert_eq!(bits(&k), bits(&expected_keys));
            assert_eq!(bits(&v), bits(&expected_values));
            let (q, k, v) = super::split_qkv(&[], 0, 2, prefix_keys, prefix_values);
            assert!(q.is_empty());
            assert_eq!(bits(&k), bits(prefix_keys));
            assert_eq!(bits(&v), bits(prefix_values));
        }
    }

    #[test]
    fn a_short_tail_does_not_double_the_retained_kv_capacity() {
        let model = cache_test_model();
        let prefix = model.run_prefix(&[super::BOS, 3, 4, 5, 6]);
        let states = model.run_tail_states(5, &prefix.keys, &prefix.values, None, &[3]);
        let width = 6 * model.config.n_embd;
        for row in states.keys.iter().chain(&states.values) {
            assert_eq!(row.len(), width);
            assert_eq!(row.capacity(), width, "短尾部保留了几何扩容后的 KV 容量");
        }
    }

    #[test]
    fn ordinary_resume_borrows_the_existing_kv_storage() {
        let model = cache_test_model();
        let prefix = model.run_prefix(&model.prefix_tokens("我", 4));
        let run = model.run_tail(&prefix, &model.encode("你好我很"));
        let (resumed, _) = run.resume(1, &model);
        assert_eq!(
            resumed.keys.as_ptr(),
            run.keys.as_ptr(),
            "普通续跑复制了 KV 外层数组"
        );
        for (left, right) in resumed.keys.iter().zip(&run.keys) {
            assert_eq!(left.as_ptr(), right.as_ptr(), "普通续跑复制了 key");
        }
        for (left, right) in resumed.values.iter().zip(&run.values) {
            assert_eq!(left.as_ptr(), right.as_ptr(), "普通续跑复制了 value");
        }
    }

    #[test]
    fn terminal_resume_borrows_the_existing_kv_storage() {
        let model = cache_test_model();
        let prefix = model.run_prefix(&model.prefix_tokens("我", 4));
        for text in ["你", "你好我很"] {
            let tokens = model.encode(text);
            let run = model.run_tail(&prefix, &tokens);
            let (resumed, _) = run.resume(tokens.len(), &model);
            assert_eq!(
                resumed.keys.as_ptr(),
                run.keys.as_ptr(),
                "末位置续跑复制了 KV 外层数组"
            );
            assert_eq!(resumed.values.as_ptr(), run.values.as_ptr());
            for (left, right) in resumed.keys.iter().zip(&run.keys) {
                assert_eq!(left.as_ptr(), right.as_ptr(), "末位置续跑复制了 key");
            }
            for (left, right) in resumed.values.iter().zip(&run.values) {
                assert_eq!(left.as_ptr(), right.as_ptr(), "末位置续跑复制了 value");
            }
        }
    }

    #[test]
    fn borrowed_resume_matches_a_truncated_owned_prefix() {
        let model = cache_test_model();
        let prefix = model.run_prefix(&model.prefix_tokens("我", 4));
        let run = model.run_tail(&prefix, &model.encode("你好我很"));
        for taken in 1..=run.cumulative.len() {
            let (resumed, carried) = run.resume(taken, &model);
            let width = resumed.length * model.config.n_embd;
            let owned = super::Prefix {
                length: resumed.length,
                keys: std::borrow::Cow::Owned(
                    run.keys
                        .iter()
                        .enumerate()
                        .map(|(layer, row)| {
                            if taken > run.hidden.len() {
                                [row.as_slice(), &run.terminal.get().unwrap().keys[layer]].concat()
                            } else {
                                row[..width].to_vec()
                            }
                        })
                        .collect(),
                ),
                values: std::borrow::Cow::Owned(
                    run.values
                        .iter()
                        .enumerate()
                        .map(|(layer, row)| {
                            if taken > run.hidden.len() {
                                [row.as_slice(), &run.terminal.get().unwrap().values[layer]]
                                    .concat()
                            } else {
                                row[..width].to_vec()
                            }
                        })
                        .collect(),
                ),
                last_hidden: resumed.last_hidden.clone(),
                terminal: None,
            };
            assert_eq!(carried.to_bits(), run.cumulative[taken - 1].to_bits());
            for text in ["吗", "吗你", "吗你好"] {
                let tail = model.encode(text);
                let actual = model.run_tail(&resumed, &tail);
                let expected = model.run_tail(&owned, &tail);
                assert_eq!(actual.base, expected.base);
                assert_eq!(actual.pending, expected.pending);
                assert_eq!(actual.keys, expected.keys);
                assert_eq!(actual.values, expected.values);
                assert_eq!(actual.hidden.len(), expected.hidden.len());
                assert_eq!(
                    actual
                        .cumulative
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .cumulative
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                for (left, right) in actual.hidden.iter().zip(&expected.hidden) {
                    assert_eq!(left.values, right.values);
                }
            }
        }
    }

    #[test]
    fn a_single_character_tail_can_be_extended_without_an_eager_state() {
        let model = cache_test_model();
        let mut reranker = super::Reranker::new(model.clone());
        let count = || super::FORWARD_POSITIONS.with(std::cell::Cell::get);
        let before = count();
        reranker.log_probabilities("我", &["你", "我"]);
        assert_eq!(count() - before, 2, "单字候选只需要上文前向");
        let before = count();
        let scores = reranker.log_probabilities("我", &["你好", "你吗"]);
        assert_eq!(count() - before, 1, "没有已计算候选位置时也应共享末字补算");
        let fresh = model.score("我", &["你好", "你吗"]);
        assert_eq!(scores, fresh);
    }

    #[test]
    fn identical_candidates_keep_the_completed_run() {
        let mut reranker = super::Reranker::new(cache_test_model());
        let before = reranker.log_probabilities("我", &["你好", "你吗"]);
        let retained = std::sync::Arc::downgrade(&reranker.resume[0].1);
        let allocation = reranker.resume[0].1.cumulative.as_ptr();
        let hidden = reranker.resume[0].1.hidden[0].values.as_ptr();
        // 重排、增删和重复候选不能消耗其它候选还要借用的旧状态。
        let after = reranker.log_probabilities("我", &["你吗我", "你好", "你好"]);
        assert_eq!(before[0].to_bits(), after[1].to_bits());
        assert_eq!(after[1].to_bits(), after[2].to_bits());
        for index in [1, 2] {
            assert_eq!(
                allocation,
                reranker.resume[index].1.cumulative.as_ptr(),
                "完全命中仍重新创建了累计分"
            );
            assert_eq!(
                hidden,
                reranker.resume[index].1.hidden[0].values.as_ptr(),
                "完全命中仍重新计算或复制了隐藏状态"
            );
        }
        assert_eq!(reranker.resume.len(), 3);
        reranker.log_probabilities("我", &["你好"]);
        assert_eq!(reranker.resume.len(), 1);
        reranker.log_probabilities("我", &[]);
        assert!(reranker.resume.is_empty());
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn fresh_score_moves_encoded_tail_without_copying_it() {
        let mut reranker = super::Reranker::new(cache_test_model());
        super::LAST_ENCODED_POINTER.with(|pointer| pointer.set(0));
        reranker.log_probabilities("我", &["你好"]);
        let encoded = super::LAST_ENCODED_POINTER.with(std::cell::Cell::get);
        assert_eq!(
            encoded,
            reranker.resume[0].0.as_ptr() as usize,
            "首次评分仍复制了编码 token 缓冲"
        );
    }

    #[test]
    fn reranker_reuses_the_resume_buffer_capacity() {
        let mut reranker = super::Reranker::new(cache_test_model());
        reranker.log_probabilities("我", &["你好", "你吗"]);
        reranker.log_probabilities("我", &["你好我", "你吗我"]);
        let spare = reranker.resume_scratch.as_ptr();
        assert!(reranker.resume_scratch.capacity() >= reranker.resume.len());
        assert!(reranker.resume_scratch.capacity() >= 2);
        reranker.log_probabilities("我", &["你好很", "你吗很"]);
        assert_eq!(reranker.resume.as_ptr(), spare);
    }

    #[test]
    fn best_where_reuses_the_score_buffer_capacity() {
        let mut reranker = super::Reranker::new(cache_test_model());
        let facts = answering(&[true, true]);
        reranker.best_where("我", &["你好", "你吗"], &facts);
        let spare = reranker.scores_scratch.as_ptr();
        assert!(reranker.scores_scratch.capacity() >= 2);
        reranker.best_where("我", &["你很", "你我"], &facts);
        assert_eq!(reranker.scores_scratch.as_ptr(), spare);
    }

    #[test]
    fn growing_candidate_lists_keep_score_storage_within_the_comparison_budget() {
        let mut reranker = super::Reranker::new(cache_test_model());
        let texts = ["你好"; 9];
        let facts = answering(&[true; 9]);
        for count in 2..=texts.len() {
            reranker.best_where("我", &texts[..count], &facts);
            assert!(
                reranker.scores_scratch.capacity() <= texts.len(),
                "{count} candidates retain capacity {}",
                reranker.scores_scratch.capacity()
            );
        }
    }

    #[test]
    fn exact_reuse_moves_encoded_tail_without_copying_it() {
        let mut reranker = super::Reranker::new(cache_test_model());
        reranker.log_probabilities("我", &["你好"]);
        super::LAST_ENCODED_POINTER.with(|pointer| pointer.set(0));
        reranker.log_probabilities("我", &["你好"]);
        let encoded = super::LAST_ENCODED_POINTER.with(std::cell::Cell::get);
        assert_eq!(
            encoded,
            reranker.resume[0].0.as_ptr() as usize,
            "复用候选仍复制了编码 token 缓冲"
        );
    }

    #[test]
    fn overlong_candidates_keep_only_the_model_window() {
        let mut reranker = super::Reranker::new(cache_test_model());
        for text in ["你", "A", "🎹"].map(|unit| unit.repeat(65_536)) {
            for _ in 0..2 {
                reranker.log_probabilities("我", &[&text]);
                let window = reranker.model.config.context - 1;
                assert_eq!(reranker.resume[0].0.len(), window);
                assert!(
                    reranker.resume[0].0.capacity() <= 2 * window,
                    "候选缓存仍保留窗口之外的编码容量：{}",
                    reranker.resume[0].0.capacity()
                );
                let encoded = super::LAST_ENCODED_LENGTH.with(std::cell::Cell::get);
                assert_eq!(encoded, window, "仍编码了不会参与评分的尾部");
            }
        }
    }

    #[test]
    fn shared_prefix_normalization_is_computed_once() {
        let mut reranker = super::Reranker::new(cache_test_model());
        let count = || super::VOCABULARY_PROJECTIONS.with(std::cell::Cell::get);
        let before = count();
        reranker.log_probabilities("我", &["你好", "你吗"]);
        // 上文和同批共享的首字状态各投影一次。
        assert_eq!(count() - before, 2, "同一上文或同批前缀重复投影词表");

        let before = count();
        reranker.log_probabilities("我", &["你好我", "你好很"]);
        assert_eq!(count() - before, 1, "同一续跑位置重复投影词表");

        let before = count();
        reranker.log_probabilities("我", &["你很", "你我"]);
        assert_eq!(count() - before, 0, "已知归一化常数的共享位置仍重复计算");

        let before = count();
        reranker.log_probabilities("很", &["你好", "你吗"]);
        assert_eq!(count() - before, 2, "上下文改变后必须使用新状态");
    }

    #[test]
    fn cached_normalization_matches_full_distribution_bits() {
        let model = cache_test_model();
        for context in ["", "我", "你好我很"] {
            let prefix = model.run_prefix(&model.prefix_tokens(context, 4));
            let run = model.run_tail(&prefix, &[3, 4, 5, 6]);
            assert!(run.terminal.get().is_none(), "未续跑的末位置不应提前计算");
            for state in std::iter::once(&prefix.last_hidden).chain(&run.hidden) {
                let full = model.distribution(&state.values);
                for token in 0..model.config.vocab + 3 {
                    let expected = full[token.min(model.config.vocab - 1)];
                    assert_eq!(
                        model.log_probability(state, token as u32).to_bits(),
                        expected.to_bits(),
                        "单 token 概率与完整分布不同"
                    );
                }
            }
        }
    }

    #[test]
    fn reranker_remains_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<super::Reranker>();
    }

    /// `answers_key` per candidate, with no candidate an exact dictionary hit.
    fn answering(flags: &[bool]) -> impl Fn(usize) -> CandidateFacts + '_ {
        move |index| CandidateFacts {
            answers_key: flags[index],
            trusted_dictionary_hit: false,
        }
    }

    #[test]
    fn only_candidates_answering_the_key_are_compared() {
        // A prefix and a completion sit among the full answers and stay out of the comparison.
        let flags = [true, false, true, false, true];
        assert_eq!(
            comparable(5, &answering(&flags)).unwrap().as_slice(),
            &[0, 2, 4]
        );
    }

    #[test]
    fn differing_character_counts_are_compared_when_both_answer_the_key() {
        // The case the old length proxy dropped: `xian` reads as 现 or as 西安, both consuming the
        // key. Nothing here looks at the text, which is the point — length is not the criterion.
        assert_eq!(
            comparable(2, &answering(&[true, true])).unwrap().as_slice(),
            &[0, 1]
        );
    }

    #[test]
    fn comparable_indices_use_fixed_storage() {
        let result = comparable(3, &answering(&[true, false, true])).unwrap();
        assert_eq!(
            std::mem::size_of_val(&result),
            std::mem::size_of::<Compared>()
        );
    }

    #[test]
    fn best_where_keeps_original_candidate_indices() {
        let model = cache_test_model();
        let texts = ["吗我", "你吗", "你好", "你很"];
        let considered = [texts[0], texts[2], texts[3]];
        let expected_scores = super::SentenceModel::score(&model, "我", &considered);
        let expected = expected_scores
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(position, _)| [0, 2, 3][position])
            .expect("at least one comparable candidate");

        let mut reranker = super::Reranker::new(model);
        let actual = reranker.best_where("我", &texts, answering(&[true, false, true, true]));
        assert_eq!(actual, (expected != 0).then_some(expected));
    }

    #[test]
    fn a_trusted_dictionary_hit_in_the_lead_declines_the_decision() {
        let facts = |index: usize| CandidateFacts {
            answers_key: true,
            trusted_dictionary_hit: index == 0,
        };
        assert_eq!(comparable(3, &facts), None);
        // The same hit anywhere but the lead is only another candidate; it does not veto.
        let trailing = |index: usize| CandidateFacts {
            answers_key: true,
            trusted_dictionary_hit: index == 2,
        };
        assert_eq!(comparable(3, &trailing).unwrap().as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn a_leader_that_does_not_answer_the_key_declines_the_decision() {
        // Promoting over a prefix would compare readings of two different keys.
        assert_eq!(comparable(3, &answering(&[false, true, true])), None);
    }

    #[test]
    fn fewer_than_two_comparable_candidates_declines_the_decision() {
        assert_eq!(comparable(0, &answering(&[])), None);
        assert_eq!(comparable(1, &answering(&[true])), None);
        assert_eq!(comparable(3, &answering(&[true, false, false])), None);
    }

    #[test]
    fn the_comparison_is_capped_and_keeps_the_leader() {
        let flags = [true; 12];
        let considered = comparable(12, &answering(&flags)).expect("all answer the key");
        assert_eq!(considered.as_slice().len(), super::COMPARED);
        assert_eq!(considered.as_slice()[0], 0);
    }

    #[test]
    fn gate_follows_the_leading_candidate_only() {
        // Database and UserDatabase lead: the dictionary knows the whole key, leave it alone.
        assert!(!should_rerank(&[0, 8, 9]));
        assert!(!should_rerank(&[1, 8]));
        // Lattice or fallback leads: no frequency evidence, so the model is worth consulting.
        assert!(should_rerank(&[8, 0, 0]));
        assert!(should_rerank(&[9, 0]));
        // Nothing to reorder.
        assert!(!should_rerank(&[]));
    }

    #[test]
    fn erf_matches_known_values() {
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
        assert!((erf(0.5) - 0.520_499_9).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-6);
        assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-6);
        assert!((erf(3.0) - 0.999_977_9).abs() < 1e-6);
    }

    #[test]
    fn gelu_matches_torch_reference() {
        // Values taken from torch.nn.functional.gelu with its default exact formulation.
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.841_344_8).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_25).abs() < 1e-6);
        assert!((gelu(2.0) - 1.954_499_7).abs() < 1e-6);
    }
}

#[cfg(test)]
mod feed_forward_scratch_tests;

#[cfg(test)]
mod prefix_final_row_tests;

#[cfg(test)]
mod terminal_prefix_tests;

#[cfg(test)]
mod embedding_tests;
