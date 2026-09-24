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

use std::collections::HashMap;
use std::sync::Arc;

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

    /// One row as `f32`, for the token table, which is read as an embedding as well as multiplied
    /// as an output projection. A row is `n_embd` values, so materializing it costs nothing.
    fn row(&self, index: usize, width: usize) -> Vec<f32> {
        let span = index * width..(index + 1) * width;
        match &self.inner {
            Tensor::Float(values) => values[span].to_vec(),
            Tensor::Quantized { values, scales } => {
                let scale = scales[index];
                values[span].iter().map(|&v| f32::from(v) * scale).collect()
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

const UNK: u32 = 1;
const BOS: u32 = 2;

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

/// The per-layer keys and values for a prefix, so that scoring several candidates against the same
/// committed text runs the prefix once instead of once per candidate.
struct Prefix {
    length: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// The normalized hidden state at the last prefix position. Position i predicts token i + 1, so
    /// this is what predicts a candidate's opening character, and without it that character would
    /// go unscored while every later one was counted.
    last_hidden: Vec<f32>,
}

/// Everything one candidate's run passed through, so that a later run agreeing with part of it can
/// start from where they stop agreeing instead of from the beginning.
struct TailRun {
    /// How many prefix positions sit in front of `keys` and `values`, which hold the prefix's
    /// entries followed by this run's.
    base: usize,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// Normalized hidden state after each of this run's positions, which is what predicts the next.
    hidden: Vec<Vec<f32>>,
    /// Log-probability of the run's first `i + 1` tokens, summed left to right.
    cumulative: Vec<f32>,
}

impl TailRun {
    /// The state after `taken` of this run's tokens, as a prefix a continuation can run against.
    ///
    /// The keys and values are truncated rather than rebuilt: position `taken`'s cache is the
    /// leading part of the one already held.
    fn resume(&self, taken: usize, n_embd: usize) -> (Prefix, f32) {
        let width = (self.base + taken) * n_embd;
        (
            Prefix {
                length: self.base + taken,
                keys: self.keys.iter().map(|key| key[..width].to_vec()).collect(),
                values: self
                    .values
                    .iter()
                    .map(|value| value[..width].to_vec())
                    .collect(),
                last_hidden: self.hidden[taken - 1].clone(),
            },
            self.cumulative[taken - 1],
        )
    }

    /// This run's leading `taken` positions followed by `rest`, as one run over the whole
    /// candidate, so the next keystroke can resume anywhere along it.
    fn joined(&self, taken: usize, carried: f32, rest: TailRun) -> TailRun {
        TailRun {
            base: self.base,
            keys: rest.keys,
            values: rest.values,
            hidden: self.hidden[..taken]
                .iter()
                .cloned()
                .chain(rest.hidden)
                .collect(),
            cumulative: self.cumulative[..taken]
                .iter()
                .copied()
                .chain(rest.cumulative.iter().map(|value| carried + value))
                .collect(),
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
        expect("tok.weight", self.token.len(), vocab * n_embd)?;
        expect("pos.weight", self.position.len(), context * n_embd)?;
        expect("ln_f.weight", self.final_weight.len(), n_embd)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            expect(
                &format!("blocks.{layer}.qkv"),
                block.qkv_weight.len(),
                3 * n_embd * n_embd,
            )?;
            expect(
                &format!("blocks.{layer}.proj"),
                block.proj_weight.len(),
                n_embd * n_embd,
            )?;
            expect(
                &format!("blocks.{layer}.fc"),
                block.fc_weight.len(),
                4 * n_embd * n_embd,
            )?;
            expect(
                &format!("blocks.{layer}.out"),
                block.out_weight.len(),
                4 * n_embd * n_embd,
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
        text.chars()
            .map(|ch| self.index.get(&ch).copied().unwrap_or(UNK))
            .collect()
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

    fn score_against(&self, prefix: &Prefix, candidates: &[&str]) -> Vec<f32> {
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
        let n_embd = self.config.n_embd;
        let mut x = vec![0.0; tokens.len() * n_embd];
        for (step, &token) in tokens.iter().enumerate() {
            let token = (token as usize).min(self.config.vocab - 1);
            let position = (offset + step).min(self.config.context - 1);
            let row = &mut x[step * n_embd..(step + 1) * n_embd];
            let token_row = self.token.row(token, n_embd);
            let position_row = &self.position[position * n_embd..(position + 1) * n_embd];
            for ((out, &t), &p) in row.iter_mut().zip(&token_row).zip(position_row) {
                *out = t + p;
            }
        }
        x
    }

    /// Runs the prefix and keeps every layer's keys and values.
    fn run_prefix(&self, tokens: &[u32]) -> Prefix {
        let n_embd = self.config.n_embd;
        let mut x = self.embed(tokens, 0);
        let mut keys = Vec::with_capacity(self.blocks.len());
        let mut values = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let normed = layer_norm(&x, &block.ln1_weight, &block.ln1_bias, n_embd);
            let qkv = linear(
                &normed,
                &block.qkv_weight,
                &block.qkv_bias,
                n_embd,
                3 * n_embd,
            );
            let (q, k, v) = split_qkv(&qkv, tokens.len(), n_embd);
            let attended = self.attention(&q, &k, &v, tokens.len(), 0);
            let projected = linear(
                &attended,
                &block.proj_weight,
                &block.proj_bias,
                n_embd,
                n_embd,
            );
            for (slot, value) in x.iter_mut().zip(projected) {
                *slot += value;
            }
            self.feed_forward(&mut x, block);
            keys.push(k);
            values.push(v);
        }
        let last = (tokens.len() - 1) * n_embd;
        let last_hidden = layer_norm(&x[last..], &self.final_weight, &self.final_bias, n_embd);
        Prefix {
            length: tokens.len(),
            keys,
            values,
            last_hidden,
        }
    }

    /// Runs a candidate against a cached prefix and returns its total log-probability.
    fn score_tail(&self, prefix: &Prefix, tail: &[u32]) -> f32 {
        let run = self.run_tail(prefix, tail);
        run.cumulative[run.cumulative.len() - 1]
    }

    /// The same run, keeping everything it passed through rather than only the total.
    ///
    /// `score_tail` builds the extended keys and values in order to attend over them and then drops
    /// them, along with the hidden state at every position and the running sum. Those are what a
    /// caller needs to stop part-way through a candidate and resume somewhere else, which is the
    /// whole point: consecutive keystrokes re-score candidates that mostly repeat the previous
    /// keystroke's, and the repeated part need not be run twice.
    ///
    /// Keeping every position rather than only the last one is what makes partial agreement usable.
    /// Candidates typically share a leading stretch and then diverge — the decoder revises the last
    /// character or two as a syllable completes — so a resume point that only exists at the end of
    /// a candidate is one almost nothing matches. Position-wise, the shared stretch is reusable
    /// however far it runs.
    ///
    /// The keys and values are stored whole and truncated when resumed: position `k`'s cache is a
    /// prefix of the full one, so a shorter resume costs a slice rather than a separate copy. Cost
    /// is therefore linear in the candidate's length, not quadratic.
    ///
    /// Grouping the sum differently is the one thing resuming does change. `(a + b) + c` and
    /// `a + (b + c)` are not the same f32, so a resumed score can differ from a single-pass score
    /// in the last place. The eval covers both paths rather than a unit test asserting equality.
    fn run_tail(&self, prefix: &Prefix, tail: &[u32]) -> TailRun {
        let n_embd = self.config.n_embd;
        let mut x = self.embed(tail, prefix.length);
        let mut keys_out = Vec::with_capacity(self.blocks.len());
        let mut values_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let normed = layer_norm(&x, &block.ln1_weight, &block.ln1_bias, n_embd);
            let qkv = linear(
                &normed,
                &block.qkv_weight,
                &block.qkv_bias,
                n_embd,
                3 * n_embd,
            );
            let (q, k, v) = split_qkv(&qkv, tail.len(), n_embd);
            let mut keys = prefix.keys[layer].clone();
            let mut values = prefix.values[layer].clone();
            keys.extend_from_slice(&k);
            values.extend_from_slice(&v);
            let attended = self.attention(&q, &keys, &values, tail.len(), prefix.length);
            let projected = linear(
                &attended,
                &block.proj_weight,
                &block.proj_bias,
                n_embd,
                n_embd,
            );
            for (slot, value) in x.iter_mut().zip(projected) {
                *slot += value;
            }
            self.feed_forward(&mut x, block);
            keys_out.push(keys);
            values_out.push(values);
        }
        let normed = layer_norm(&x, &self.final_weight, &self.final_bias, n_embd);

        // Position i predicts token i + 1. The opening character is therefore predicted by the last
        // prefix position, and every later one by the tail position before it.
        let mut running = self.log_probability(&prefix.last_hidden, tail[0]);
        let mut cumulative = Vec::with_capacity(tail.len());
        cumulative.push(running);
        let mut hidden = Vec::with_capacity(tail.len());
        for step in 0..tail.len() {
            let row = normed[step * n_embd..(step + 1) * n_embd].to_vec();
            if step + 1 < tail.len() {
                running += self.log_probability(&row, tail[step + 1]);
                cumulative.push(running);
            }
            hidden.push(row);
        }
        TailRun {
            base: prefix.length,
            keys: keys_out,
            values: values_out,
            hidden,
            cumulative,
        }
    }

    fn feed_forward(&self, x: &mut [f32], block: &Block) {
        let n_embd = self.config.n_embd;
        let normed = layer_norm(x, &block.ln2_weight, &block.ln2_bias, n_embd);
        let mut hidden = linear(
            &normed,
            &block.fc_weight,
            &block.fc_bias,
            n_embd,
            4 * n_embd,
        );
        for value in &mut hidden {
            *value = gelu(*value);
        }
        let out = linear(
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

    /// log P(token) under the tied output projection, computed as the logit minus the log of the
    /// summed exponentials. The full vocabulary is needed for the denominator, which is why this is
    /// the most expensive step per scored character.
    fn log_probability(&self, hidden: &[f32], token: u32) -> f32 {
        let token = (token as usize).min(self.config.vocab - 1);
        self.distribution(hidden)[token]
    }

    /// log P over the whole vocabulary for the state at `hidden`.
    fn distribution(&self, hidden: &[f32]) -> Vec<f32> {
        let n_embd = self.config.n_embd;
        let mut logits = Vec::with_capacity(self.config.vocab);
        let mut highest = f32::NEG_INFINITY;
        for row in 0..self.config.vocab {
            let logit = self.token.dot_row(hidden, row, n_embd);
            highest = highest.max(logit);
            logits.push(logit);
        }
        let sum: f32 = logits.iter().map(|logit| (logit - highest).exp()).sum();
        let offset = highest + sum.ln();
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
        self.distribution(&prefix.last_hidden)
    }

    /// The identifier this model uses for `character`, when it has one.
    pub fn token(&self, character: char) -> Option<u32> {
        self.index.get(&character).copied()
    }

    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], rows: usize, offset: usize) -> Vec<f32> {
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
    cached: Option<(Vec<u32>, Prefix)>,
    /// What the previous decision ended up knowing, one entry per candidate it scored: the tokens
    /// it ran, the state it left the model in, and the log-probability accumulated over them.
    ///
    /// A keystroke rescores candidates that are mostly the previous keystroke's candidates with a
    /// character added, so most of each candidate has already been run. Measured on the sentence
    /// eval driven through the real runtime, 61% of the characters scored are a prefix of
    /// something the previous keystroke scored. Keeping only the previous decision bounds this to
    /// one entry per candidate; keeping more would grow without a ceiling for reuse that was not
    /// measured to be there.
    resume: Vec<(Vec<u32>, TailRun)>,
}

impl Reranker {
    pub fn new(model: Arc<SentenceModel>) -> Self {
        Self {
            model,
            cached: None,
            resume: Vec::new(),
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
        let subset: Vec<&str> = considered.iter().map(|&index| texts[index]).collect();
        let scores = self.score(context, &subset);
        let mut best = 0;
        for (index, score) in scores.iter().enumerate() {
            if *score > scores[best] {
                best = index;
            }
        }
        (best != 0).then(|| considered[best])
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
        let longest = candidates
            .iter()
            .map(|text| text.chars().count())
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
            // Every resume point hangs off the old prefix and means nothing against a new one.
            self.resume.clear();
        }
        let (_, prefix) = self.cached.as_ref().expect("prefix was just computed");
        let n_embd = self.model.config.n_embd;

        let mut next = Vec::with_capacity(candidates.len());
        let mut scores = Vec::with_capacity(candidates.len());
        for text in candidates {
            let tail = self.model.encode(text);
            if tail.is_empty() {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let limit = self
                .model
                .config
                .context
                .saturating_sub(prefix.length)
                .max(1);
            let tail = &tail[..tail.len().min(limit)];

            // How far this candidate agrees with the furthest-agreeing run from last time. One
            // short of `tail.len()` at most: a resume has to leave a token to run, both because
            // the run needs one and because a candidate with nothing left to score has no term.
            let shared = self
                .resume
                .iter()
                .map(|(tokens, _)| {
                    let agreed = tokens
                        .iter()
                        .zip(tail)
                        .take_while(|(left, right)| left == right)
                        .count();
                    agreed.min(tail.len() - 1)
                })
                .enumerate()
                .max_by_key(|(_, agreed)| *agreed);

            let run = match shared {
                Some((index, agreed)) if agreed > 0 => {
                    let (from, carried) = self.resume[index].1.resume(agreed, n_embd);
                    let rest = self.model.run_tail(&from, &tail[agreed..]);
                    self.resume[index].1.joined(agreed, carried, rest)
                }
                _ => self.model.run_tail(prefix, tail),
            };
            let total = run.cumulative[run.cumulative.len() - 1];
            scores.push(total / tail.len() as f32);
            next.push((tail.to_vec(), run));
        }
        self.resume = next;
        scores
    }
}

/// Which candidates may be compared, or `None` when the decision should be declined.
///
/// Separate from scoring and free of the model, because this is the part that decides what the
/// measurements mean: every published accuracy number for this crate is a consequence of which
/// candidates were allowed into the comparison, so it is worth being able to test on its own.
fn comparable(count: usize, facts: &impl Fn(usize) -> CandidateFacts) -> Option<Vec<usize>> {
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
    let considered: Vec<usize> = (0..count)
        .filter(|&index| facts(index).answers_key)
        .take(COMPARED)
        .collect();
    (considered.len() >= 2).then_some(considered)
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
fn linear(x: &[f32], w: &Matrix, bias: &[f32], inputs: usize, outputs: usize) -> Vec<f32> {
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

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], width: usize) -> Vec<f32> {
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

fn split_qkv(qkv: &[f32], rows: usize, n_embd: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; rows * n_embd];
    let mut k = vec![0.0; rows * n_embd];
    let mut v = vec![0.0; rows * n_embd];
    for row in 0..rows {
        let source = &qkv[row * 3 * n_embd..(row + 1) * 3 * n_embd];
        q[row * n_embd..(row + 1) * n_embd].copy_from_slice(&source[..n_embd]);
        k[row * n_embd..(row + 1) * n_embd].copy_from_slice(&source[n_embd..2 * n_embd]);
        v[row * n_embd..(row + 1) * n_embd].copy_from_slice(&source[2 * n_embd..]);
    }
    (q, k, v)
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
    use super::{comparable, erf, gelu, should_rerank, CandidateFacts};

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
        assert_eq!(comparable(5, &answering(&flags)), Some(vec![0, 2, 4]));
    }

    #[test]
    fn differing_character_counts_are_compared_when_both_answer_the_key() {
        // The case the old length proxy dropped: `xian` reads as 现 or as 西安, both consuming the
        // key. Nothing here looks at the text, which is the point — length is not the criterion.
        assert_eq!(comparable(2, &answering(&[true, true])), Some(vec![0, 1]));
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
        assert_eq!(comparable(3, &trailing), Some(vec![0, 1, 2]));
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
        assert_eq!(considered.len(), super::COMPARED);
        assert_eq!(considered[0], 0);
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
