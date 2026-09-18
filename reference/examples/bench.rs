//! Measures how a reranking decision's cost scales, which is what decides whether the model can run
//! on the keystroke path or has to be moved off it.
//!
//! An earlier version of this timed nine candidates of one fixed length, five runs each, and
//! reported the mean. That misses the variable that decides the answer. Cost grows with how many
//! characters are being scored, so the expensive decisions are the ones late in a long sentence —
//! and a mean of five says nothing about how bad they get. Measured on the input method this model
//! came from, the mean extra cost per keystroke was 7.9ms while a quarter of keystrokes went over a
//! 60 Hz frame and the worst took 51ms. Both numbers came from the same weights. Only one of them
//! answers "can this run on the keystroke path".
//!
//! Two things are reported: the cost of a single decision across the shapes a decision can take,
//! and the cost of typing one sentence end to end, which is the sum of the decisions a user
//! actually triggers.
//!
//! Candidates are prefixes of the evaluation set's sentences. Cost depends on the shape of the
//! request — how many candidates, how many characters each — and not on which characters they are,
//! so real text at a controlled length measures the same thing as invented text without inventing
//! any.
//!
//! usage: bench <model.safetensors> [sentences.tsv] [--budget-ms N]

use std::time::{Duration, Instant};

use chinese_ime_lm::SentenceModel;

/// Candidate counts worth measuring. One is the floor — a host that reranks a single candidate is
/// paying the model to confirm the only answer — and nine is a full candidate page.
const COUNTS: [usize; 4] = [1, 3, 5, 9];

/// Character counts worth measuring, up to a long sentence. A user commits long before 15
/// characters, but the tail is the part that decides whether this fits on the keystroke path.
const LENGTHS: [usize; 8] = [1, 2, 4, 6, 8, 10, 12, 15];

/// Decisions timed per shape. Enough that a percentile means something, few enough that the whole
/// sweep stays under a minute for a small model.
const RUNS: usize = 40;

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Nearest-rank percentile, so every number reported is a decision that actually happened.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// Times `RUNS` decisions of one shape and returns them sorted, in milliseconds.
fn measure(model: &SentenceModel, context: &str, candidates: &[&str]) -> Vec<f64> {
    // One untimed pass so the first-touch page faults for these weights land outside the samples.
    let _ = model.score(context, candidates);
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        let _ = model.score(context, candidates);
        samples.push(ms(started.elapsed()));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples
}

/// Sentences to cut candidates from, read from the evaluation set when one is given.
///
/// The fallback is deliberately small and obviously fake. A benchmark that silently substitutes
/// its own text when the real set is missing reports a number for something other than what it
/// claims to have measured, so this says so on the line above the table.
fn sentences(path: Option<&str>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(vec![
            "我今天去上班".into(),
            "明天下午开会讨论这个问题".into(),
            "这份文件需要经理签字".into(),
            "我们应该把这个问题解决掉".into(),
        ]);
    };
    let text = std::fs::read_to_string(path)?;
    let mut sentences = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() || line.starts_with("id\t") {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if let Some(gold) = fields.get(2) {
            if !gold.is_empty() {
                sentences.push((*gold).to_string());
            }
        }
    }
    if sentences.is_empty() {
        return Err(format!("{path}: no sentences").into());
    }
    Ok(sentences)
}

/// Distinct prefixes of `length` characters, as a candidate list of `count` equal-length strings.
///
/// Equal length is not a convenience. Summed log probability favours shorter text, so a mixed-length
/// list is not a comparison at all — `docs/format.md` says why, and a benchmark that ignored it
/// would be timing a request no caller should ever make.
/// Returns the list and how many entries had to be repeated to reach `count`.
fn candidates(pool: &[String], count: usize, length: usize) -> (Vec<String>, usize) {
    let mut distinct: Vec<String> = Vec::new();
    for sentence in pool {
        if sentence.chars().count() < length {
            continue;
        }
        let prefix: String = sentence.chars().take(length).collect();
        if !distinct.contains(&prefix) {
            distinct.push(prefix);
        }
        if distinct.len() == count {
            break;
        }
    }
    if distinct.is_empty() {
        return (Vec::new(), 0);
    }
    // Short sets repeat rather than shrink. A row labelled "9 candidates" has to have timed nine of
    // them, or the table reads as though the model got faster at longer lengths when what actually
    // happened is that the pool ran out of sentences that long. How many were repeats is returned
    // rather than swallowed, because it is the one way this table can mislead.
    let repeats = count.saturating_sub(distinct.len());
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(distinct[index % distinct.len()].clone());
    }
    (out, repeats)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut budget_ms: Option<f64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--budget-ms" => {
                budget_ms = Some(
                    args.next()
                        .ok_or("--budget-ms needs a value")?
                        .parse::<f64>()?,
                )
            }
            other => positional.push(other.to_string()),
        }
    }
    let path = positional
        .first()
        .ok_or("usage: bench <model.safetensors> [sentences.tsv] [--budget-ms N]")?;
    let pool = sentences(positional.get(1).map(String::as_str))?;

    let started = Instant::now();
    let bytes = std::fs::read(path)?;
    let model = SentenceModel::load(&bytes)?;
    println!(
        "load {:?} for {:.1} MB",
        started.elapsed(),
        bytes.len() as f64 / 1e6
    );
    println!("attribution: {}", model.attribution());
    println!(
        "{} sentences{}",
        pool.len(),
        match positional.get(1) {
            Some(path) => format!(" from {path}"),
            None => " from the built-in fallback, not the evaluation set".to_string(),
        }
    );

    println!("\none decision, milliseconds (p50 / p95 over {RUNS} runs):");
    print!("{:>12}", "chars:");
    for length in LENGTHS {
        print!("{length:>14}");
    }
    println!();
    let mut padded = false;
    for count in COUNTS {
        print!("{count:>10} cand");
        for length in LENGTHS {
            let (texts, repeats) = candidates(&pool, count, length);
            padded |= repeats > 0;
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let samples = measure(&model, "", &refs);
            print!(
                "{:>7.2} /{:>6.2}",
                percentile(&samples, 50.0),
                percentile(&samples, 95.0)
            );
        }
        println!();
    }
    if padded {
        println!(
            "some cells repeated candidates: the pool has too few sentences that long. Cost depends\non the shape of the request, so the timings hold, but the text in those cells is not 9 distinct\nsentences."
        );
    }

    // What a host actually pays. Typing an n-character sentence does not cost one decision at
    // length n; it costs a decision at every length up to n, because the candidate list is rescored
    // from scratch on each keystroke. That the totals below grow faster than the single-decision row
    // is the whole point: the per-keystroke cost is not flat, and the last keystroke of a sentence
    // is the most expensive one in it.
    println!("\ntyping one sentence with 9 candidates, rescored every keystroke:");
    println!(
        "{:>10}  {:>12}  {:>14}  {:>14}",
        "chars", "total ms", "worst key ms", "mean key ms"
    );
    let mut worst_overall: f64 = 0.0;
    for length in LENGTHS {
        let mut total = 0.0;
        let mut worst: f64 = 0.0;
        for prefix in 1..=length {
            let (texts, _) = candidates(&pool, 9, prefix);
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let samples = measure(&model, "", &refs);
            let typical = percentile(&samples, 50.0);
            total += typical;
            worst = worst.max(typical);
        }
        worst_overall = worst_overall.max(worst);
        println!(
            "{length:>10}  {total:>12.2}  {worst:>14.2}  {:>14.2}",
            total / length as f64
        );
    }

    if let Some(budget) = budget_ms {
        if worst_overall > budget {
            return Err(format!(
                "the slowest keystroke costs {worst_overall:.2}ms, over the {budget:.2}ms budget"
            )
            .into());
        }
        println!("\nslowest keystroke {worst_overall:.2}ms, within the {budget:.2}ms budget");
    } else {
        println!(
            "\nslowest keystroke {worst_overall:.2}ms. Pass --budget-ms to fail when it is too slow;\nwhat counts as too slow belongs to the host, not to this crate."
        );
    }

    // A ranking the reader can check by eye. A timing table proves the model is fast, not that it is
    // still answering correctly, and a benchmark that silently broke the forward pass would print
    // very good numbers.
    let check = [
        "我今天去上班",
        "握紧天趣上班",
        "我今天去上版",
        "我今天趣上班",
        "握今天去上班",
    ];
    let scores = model.score("", &check);
    let mut ranked: Vec<_> = check.iter().zip(&scores).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(a.1));
    println!("\nranking check:");
    for (text, score) in ranked {
        println!("  {score:>8.4}  {text}");
    }
    Ok(())
}
