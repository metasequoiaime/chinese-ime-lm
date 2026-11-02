//! Scores chinese-ime-lm's candidate pool with the Qingjian IME's own model code.
//!
//! This file belongs to chinese-ime-lm (`eval/open-models/`, Apache-2.0) and links nothing of
//! Qingjian's into that repository. It is meant to be copied into a Qingjian checkout as
//! `crates/qingjian-neural/examples/ime_lm_pool.rs` and built there, so that every model is run by
//! the code its own input method runs, not by a reimplementation:
//!
//!     cargo build --release -p qingjian-neural --example ime_lm_pool --features metal,zhiyi
//!
//! `metal,zhiyi` are the features the Qingjian macOS app builds `qingjian-neural` with. Without
//! `metal` the same code runs on the CPU.
//!
//! What each model is given is what the Qingjian engine gives it:
//!
//! - `tongbian` (含章·通变): `P2cScorer`'s `SentenceScorer::score`, which is `log P(candidate | typed
//!   letters)` and ignores the context. When a candidate holds a character outside the model's
//!   vocabulary it scores nothing and the engine keeps its own order; that is kept here (the pick is
//!   the pool's first candidate, and the case is reported as declined).
//! - `zhiwei` (含章·知微): `CharScorer`'s `SentenceScorer::score`, `log P(candidate | context)`,
//!   ignoring the letters. Same vocabulary rule.
//! - `zhiyi` (知意 1.4V2-Long-L12, safetensors): `ChoiceScorer` through `ChoiceModel`, as
//!   `Engine::rerank_choices` calls it: the typed units are the leading candidate's syllables
//!   aligned with the letters (`ChoiceModel::align`), and one call scores the whole pool in its
//!   order. The pick is the highest logit. The leading candidate's syllables are not in the pool,
//!   so they are recovered here: the letters are split into as many syllables of the model's unit
//!   table as the case has, and when more than one split exists, the one whose syllables the
//!   leader's characters can be read as (the model's own `readings` table) is taken.
//!
//! The context is what the Qingjian engine hands each model, at its default settings: the last
//! `RESCORE_CONTEXT_CHARS` (64) characters for `zhiwei`, which is the rescoring context, and the last
//! `ChoiceTier::default().context_chars(false)` (64, the default "balanced" tier) for `zhiyi`.
//! `tongbian` reads none.
//!
//! A case typed as initials or mixed (`style` in the pool; full pinyin when absent) has no
//! syllables of its own to split the letters into, so the leader's syllables are its characters'
//! readings (the model's `readings` table) that `ChoiceModel::align` can match against the typed
//! letters, the first such reading in the table's order where more than one gives different units.
//! When no reading of the leader aligns (`align` is greedy: `shgg` spends `sh` on 社 of 社会公共),
//! Qingjian does not rerank the page, and the case is reported as declined (the pool's first).
//! `tongbian` is `log P(candidate | complete pinyin)` and is not given abbreviated cases (the
//! adapter refuses them), since Qingjian never calls it with anything but full pinyin.
//!
//! In every case the pick is the model alone. The Qingjian engine adds its own terms (the lattice
//! score, an n-gram, a rank prior); those belong to its candidate generation and are left out, as
//! they are for every other scorer in the comparison.
//!
//! The time is one scoring call per case, the whole pool at once, after one untimed pass over all
//! cases. Each pass runs the cases in order, so a model's cache of the typed letters or the context
//! never carries over from the same case; the reported figure for a case is its median over passes.
//!
//! usage: ime_lm_pool <tongbian|zhiwei|zhiyi> <model file or directory> <pool.jsonl> [--passes N]
//! Prints one JSON object per case on stdout and a summary on stderr.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use qingjian_core::{ChoiceModel, ChoiceTier, RESCORE_CONTEXT_CHARS};
use qingjian_core::sentence::SentenceScorer;
use qingjian_neural::zhiyi::ChoiceScorer;
use qingjian_neural::{CharScorer, P2cScorer};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    id: String,
    input: String,
    gold: String,
    syllables: usize,
    #[serde(default)]
    context: String,
    #[serde(default = "full_style")]
    style: String,
    candidates: Vec<String>,
}

fn full_style() -> String {
    "full".to_owned()
}

/// The last `count` characters of `text`, as the Qingjian engine's `take_last_chars` cuts them.
fn take_last_chars(text: &str, count: usize) -> String {
    let total = text.chars().count();
    text.chars().skip(total.saturating_sub(count)).collect()
}

#[derive(Deserialize)]
struct UnifiedVocab {
    units: Vec<String>,
    readings: HashMap<String, Vec<String>>,
}

enum Scorer {
    Sentence(Box<dyn SentenceScorer>),
    Choice {
        model: ChoiceScorer,
        syllables: Vec<String>,
        readings: HashMap<String, Vec<String>>,
    },
}

/// Every way to split `keys` into exactly `count` syllables from `table`.
fn splits(keys: &str, count: usize, table: &[String]) -> Vec<Vec<String>> {
    if keys.is_empty() {
        return if count == 0 {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
    }
    if count == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for syllable in table.iter().filter(|s| keys.starts_with(s.as_str())) {
        for mut rest in splits(&keys[syllable.len()..], count - 1, table) {
            rest.insert(0, syllable.clone());
            out.push(rest);
        }
    }
    out
}

/// Readings of `characters` that consume `keys` the way `ChoiceModel::align` does (a syllable typed
/// in full, else the longest common prefix), every solution in the readings table's order.
fn abbreviated_readings(
    keys: &str,
    characters: &[String],
    readings: &HashMap<String, Vec<String>>,
    chosen: &mut Vec<String>,
    out: &mut Vec<Vec<String>>,
) {
    if out.len() >= 64 {
        return;
    }
    let Some(character) = characters.first() else {
        if keys.is_empty() {
            out.push(chosen.clone());
        }
        return;
    };
    if keys.is_empty() {
        return;
    }
    for reading in readings.get(character).map(Vec::as_slice).unwrap_or_default() {
        let consumed = if keys.starts_with(reading.as_str()) {
            reading.len()
        } else {
            keys.bytes()
                .zip(reading.bytes())
                .take_while(|(a, b)| a == b)
                .count()
        };
        if consumed == 0 {
            continue;
        }
        chosen.push(reading.clone());
        abbreviated_readings(&keys[consumed..], &characters[1..], readings, chosen, out);
        chosen.pop();
    }
}

/// The leading candidate's syllables, as described in the module comment.
fn lead_syllables(
    case: &Case,
    table: &[String],
    readings: &HashMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    if case.style != "full" {
        let lead: Vec<String> = case.candidates[0].chars().map(String::from).collect();
        let mut found = Vec::new();
        abbreviated_readings(&case.input, &lead, readings, &mut Vec::new(), &mut found);
        return found.into_iter().next().ok_or_else(|| {
            format!(
                "{}: no reading of {} fits {}",
                case.id, case.candidates[0], case.input
            )
        });
    }
    let mut found = splits(&case.input, case.syllables, table);
    if found.len() > 1 {
        let lead: Vec<String> = case.candidates[0].chars().map(String::from).collect();
        found.retain(|split| {
            split.iter().zip(&lead).all(|(syllable, character)| {
                readings
                    .get(character)
                    .is_some_and(|list| list.contains(syllable))
            })
        });
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        n => Err(format!(
            "{}: {n} readings of {} fit the leader",
            case.id, case.input
        )),
    }
}

impl Scorer {
    /// The model's scores for the pool, in pool order; empty when the model declines.
    fn score(&self, case: &Case, prepared: Option<&[u32]>) -> Vec<f64> {
        let texts: Vec<&str> = case.candidates.iter().map(String::as_str).collect();
        match self {
            Scorer::Sentence(scorer) => {
                let context = take_last_chars(&case.context, RESCORE_CONTEXT_CHARS);
                scorer.score(&context, &case.input, &texts)
            }
            Scorer::Choice { model, .. } => {
                // No units: the leader's reading does not align with the letters, and the engine
                // leaves the page in its own order.
                let Some(typed) = prepared else {
                    return Vec::new();
                };
                let context =
                    take_last_chars(&case.context, ChoiceTier::default().context_chars(false));
                ChoiceModel::score(model, &context, typed, &texts)
                    .into_iter()
                    .map(f64::from)
                    .collect()
            }
        }
    }

    /// The typed units for the choice model (aligned outside the timing, as the engine does it in
    /// the keystroke callback); nothing for the others.
    fn prepare(&self, case: &Case) -> Result<Option<Vec<u32>>, String> {
        let Scorer::Choice {
            model,
            syllables,
            readings,
        } = self
        else {
            return Ok(None);
        };
        if case.style != "full" {
            // `align` takes the longest common prefix of each syllable, so `shgg` against 社会公共
            // spends `sh` on 社 and has nothing left to match 会: no reading of the leader aligns,
            // and Qingjian's `rerank_choices` leaves such a page alone. Reported as declined.
            return Ok(lead_syllables(case, syllables, readings)
                .ok()
                .and_then(|lead| ChoiceModel::align(model, &case.input, &lead)));
        }
        let lead = lead_syllables(case, syllables, readings)?;
        ChoiceModel::align(model, &case.input, &lead)
            .map(Some)
            .ok_or_else(|| format!("{}: {lead:?} does not align with {}", case.id, case.input))
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p / 100.0 * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: ime_lm_pool <tongbian|zhiwei|zhiyi> <model> <pool.jsonl> [--passes N]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(kind), Some(model), Some(pool)) = (args.first(), args.get(1), args.get(2)) else {
        return Err(usage.into());
    };
    let passes: usize = match args.get(3).map(String::as_str) {
        Some("--passes") => args.get(4).ok_or(usage)?.parse()?,
        Some(other) => return Err(format!("unknown option {other}\n{usage}").into()),
        None => 5,
    };

    let path = Path::new(model);
    let started = Instant::now();
    let scorer = match kind.as_str() {
        "tongbian" => Scorer::Sentence(Box::new(
            P2cScorer::new(CharScorer::load(path)?).ok_or("not a P2C model (no <sep>)")?,
        )),
        "zhiwei" => Scorer::Sentence(Box::new(CharScorer::load(path)?)),
        "zhiyi" => {
            let vocab: UnifiedVocab =
                serde_json::from_str(&std::fs::read_to_string(path.join("unified_vocab.json"))?)?;
            let mut syllables: Vec<String> = vocab
                .units
                .iter()
                .filter_map(|unit| unit.strip_prefix("f:").map(String::from))
                .collect();
            syllables.sort();
            Scorer::Choice {
                model: ChoiceScorer::load(path)?,
                syllables,
                readings: vocab.readings,
            }
        }
        other => return Err(format!("unknown model kind {other}\n{usage}").into()),
    };
    eprintln!(
        "{kind}: loaded in {:.0} ms",
        started.elapsed().as_secs_f64() * 1e3
    );

    let cases: Vec<Case> = std::fs::read_to_string(pool)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    if kind == "tongbian" {
        if let Some(case) = cases.iter().find(|case| case.style != "full") {
            return Err(format!(
                "{}: tongbian scores complete pinyin only, and this case is typed {}",
                case.id, case.style
            )
            .into());
        }
    }
    let prepared: Vec<Option<Vec<u32>>> = cases
        .iter()
        .map(|case| scorer.prepare(case))
        .collect::<Result<_, _>>()?;

    // Untimed: weights paged in, Metal pipelines compiled.
    for (case, units) in cases.iter().zip(&prepared) {
        let _ = scorer.score(case, units.as_deref());
    }

    let mut samples = vec![Vec::with_capacity(passes); cases.len()];
    let mut scores = vec![Vec::new(); cases.len()];
    for _ in 0..passes.max(1) {
        for (index, (case, units)) in cases.iter().zip(&prepared).enumerate() {
            let started = Instant::now();
            let scored = scorer.score(case, units.as_deref());
            samples[index].push(started.elapsed().as_secs_f64() * 1e3);
            scores[index] = scored;
        }
    }

    let (mut hits, mut declined) = (0, 0);
    let mut decisions = Vec::with_capacity(cases.len());
    for ((case, scored), times) in cases.iter().zip(&scores).zip(&mut samples) {
        let pick = if scored.len() == case.candidates.len() {
            let mut best = 0;
            for (index, score) in scored.iter().enumerate() {
                if *score > scored[best] {
                    best = index;
                }
            }
            best
        } else {
            declined += 1;
            0
        };
        hits += usize::from(case.candidates[pick] == case.gold);
        let ms = median(times);
        decisions.push(ms);
        println!(
            "{}",
            serde_json::json!({
                "id": case.id,
                "pick": pick,
                "declined": scored.len() != case.candidates.len(),
                "scores": scored,
                "ms": ms,
            })
        );
    }
    decisions.sort_by(|a, b| a.total_cmp(b));
    eprintln!(
        "{kind}: {hits}/{} hits, {declined} declined; decision p50 {:.2} ms, p95 {:.2} ms ({passes} passes)",
        cases.len(),
        percentile(&decisions, 50.0),
        percentile(&decisions, 95.0),
    );
    Ok(())
}
