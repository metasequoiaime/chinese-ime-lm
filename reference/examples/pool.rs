//! Picks one candidate per case from a fixed candidate pool, and times each pick.
//!
//! This is the scorer behind the comparison with other models in `docs/measurements.md`. Every
//! scorer there gets the same pool for each case and has to choose one candidate from it; a case
//! counts when the choice is the gold. The pool is recorded once (`eval/open-models/`), so other
//! scorers, written in other languages against other runtimes, answer the same question on the same
//! input, and their picks can be compared case by case rather than by total.
//!
//! The pick is the candidate with the highest mean log-probability per character, the first one on
//! a tie, which is what `Reranker::best_where` does once it has decided to compare. Candidates in a
//! pool all answer the whole key and have the same length, so the gate in front of that decision has
//! nothing to decline here.
//!
//! The time is one `SentenceModel::score` call over the whole pool: the cost of a decision taken
//! from scratch, with no state kept between decisions. That is the "每次按键整条重算" column of the
//! keystroke table, not the resumed path a host gets through `Reranker`. Each pass goes through the
//! cases in order and the reported figure for a case is its median over the passes.
//!
//! Prints one JSON object per case on stdout (`id`, `pick`, `scores`, `ms`) and a summary on stderr.
//!
//! usage: pool <model.safetensors> <pool.jsonl> [--passes N]

use std::time::Instant;

use chinese_ime_lm::SentenceModel;

#[derive(serde::Deserialize)]
struct Case {
    id: String,
    gold: String,
    #[serde(default)]
    context: String,
    candidates: Vec<String>,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

/// Nearest-rank percentile, so the number reported is a decision that actually happened.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p / 100.0 * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: pool <model.safetensors> <pool.jsonl> [--passes N]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = args.first().ok_or(usage)?;
    let pool_path = args.get(1).ok_or(usage)?;
    let passes: usize = match args.get(2).map(String::as_str) {
        Some("--passes") => args.get(3).ok_or(usage)?.parse()?,
        Some(other) => return Err(format!("unknown option {other}\n{usage}").into()),
        None => 5,
    };

    let model = SentenceModel::load(&std::fs::read(model_path)?)?;
    let cases: Vec<Case> = std::fs::read_to_string(pool_path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;

    // One untimed pass so the first-touch page faults for the weights land outside the samples.
    for case in &cases {
        let texts: Vec<&str> = case.candidates.iter().map(String::as_str).collect();
        let _ = model.score(&case.context, &texts);
    }

    let mut samples = vec![Vec::with_capacity(passes); cases.len()];
    let mut scores = vec![Vec::new(); cases.len()];
    for _ in 0..passes.max(1) {
        for (index, case) in cases.iter().enumerate() {
            let texts: Vec<&str> = case.candidates.iter().map(String::as_str).collect();
            let started = Instant::now();
            let scored = model.score(&case.context, &texts);
            samples[index].push(started.elapsed().as_secs_f64() * 1000.0);
            scores[index] = scored;
        }
    }

    let mut hits = 0;
    let mut decisions = Vec::with_capacity(cases.len());
    let mut per_candidate = Vec::with_capacity(cases.len());
    for ((case, scored), times) in cases.iter().zip(&scores).zip(&mut samples) {
        let mut pick = 0;
        for (index, score) in scored.iter().enumerate() {
            if *score > scored[pick] {
                pick = index;
            }
        }
        hits += usize::from(case.candidates[pick] == case.gold);
        let ms = median(times);
        decisions.push(ms);
        per_candidate.push(ms / case.candidates.len() as f64);
        println!(
            "{}",
            serde_json::json!({"id": case.id, "pick": pick, "scores": scored, "ms": ms})
        );
    }
    decisions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "{hits}/{} hits; decision p50 {:.2} ms, p95 {:.2} ms, mean {:.2} ms; per candidate p50 {:.3} ms ({passes} passes)",
        cases.len(),
        percentile(&decisions, 50.0),
        percentile(&decisions, 95.0),
        decisions.iter().sum::<f64>() / decisions.len() as f64,
        median(&mut per_candidate),
    );
    Ok(())
}
