//! Checks that a model file is one this project would publish, and says why when it is not.
//!
//! Two things this guards were real defects, not hypotheticals. The corpus attribution was a
//! hardcoded constant naming every source the pipeline can fetch, so a model trained purely on
//! permissive text claimed Chinese Wikipedia; and the licence was hardcoded to the repository's own
//! GPL-3.0, which is share-alike and would hand every adopter the obligation this project's corpus
//! policy exists to avoid. Both were fields whose only purpose is to be relied upon, and both were
//! wrong in a way nothing would have caught.
//!
//! It is run over every asset of a release, and it is also the tool to run on a model you
//! downloaded: `SECURITY.md` asks you not to load one you have not checked, and this is what
//! checking means beyond the digest.
//!
//! usage: verify <model.safetensors> [--expect-sha256 <hex>] [--expect-size <bytes>]

use std::process::ExitCode;

use chinese_ime_lm::SentenceModel;

/// Licences a published model may carry. An input method that cannot link GPL code cannot use GPL
/// weights either, so a copyleft identifier here means the file is not adoptable by most of the
/// projects this exists for — which makes publishing it a mistake rather than a choice.
const PERMISSIVE: [&str; 7] = [
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "CC-BY-4.0",
    "Unlicense",
];

/// Corpora whose licences impose share-alike. Matched against the attribution text rather than
/// against a list of sources used, because the attribution is what travels with the file and what a
/// redistributor will read.
const SHARE_ALIKE: [&str; 4] = ["Wikipedia", "MDN", "CC BY-SA", "CC-BY-SA"];

/// A sentence whose correct reading is unambiguous, and homophone confusions of it. A file can load
/// cleanly, carry perfect metadata, and still have been quantized wrong or truncated and repaired;
/// nothing in the header would show it, and the timings would look excellent.
const CHECK_SENTENCE: [&str; 5] = [
    "我今天去上班",
    "握紧天趣上班",
    "我今天去上版",
    "我今天趣上班",
    "握今天去上班",
];

fn fail(message: &str) -> ExitCode {
    eprintln!("拒绝: {message}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        return fail(
            "usage: verify <model.safetensors> [--expect-sha256 <hex>] [--expect-size <bytes>]",
        );
    };
    let mut expect_sha: Option<String> = None;
    let mut expect_size: Option<u64> = None;
    while let Some(flag) = args.next() {
        match (flag.as_str(), args.next()) {
            ("--expect-sha256", Some(value)) => expect_sha = Some(value.to_lowercase()),
            ("--expect-size", Some(value)) => match value.parse() {
                Ok(size) => expect_size = Some(size),
                Err(_) => return fail(&format!("--expect-size is not a number: {value}")),
            },
            (other, _) => return fail(&format!("unknown argument: {other}")),
        }
    }

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => return fail(&format!("{path}: {error}")),
    };
    println!("{path}: {} 字节", bytes.len());

    if let Some(expected) = expect_size {
        if bytes.len() as u64 != expected {
            return fail(&format!("体积是 {} 字节，声明的是 {expected}", bytes.len()));
        }
        println!("体积与声明一致");
    }
    if let Some(expected) = expect_sha {
        let actual = sha256(&bytes);
        if actual != expected {
            return fail(&format!("摘要是 {actual}，声明的是 {expected}"));
        }
        println!("摘要与声明一致 {actual}");
    }

    // Loading is itself most of the check: the loader validates the header bounds, every tensor's
    // offsets, the vocabulary length against the configuration, and that the channels divide into
    // the attention heads. A file that gets past it is structurally a model.
    let model = match SentenceModel::load(&bytes) {
        Ok(model) => model,
        Err(error) => return fail(&format!("加载失败: {error}")),
    };
    println!("加载成功 {model:?}");

    let license = model.license();
    if license.is_empty() {
        return fail("没有 license 字段。权重的许可必须写在文件里，沉默不等于允许");
    }
    if !PERMISSIVE.contains(&license) {
        return fail(&format!(
            "许可为 {license}，不在可发布清单内 {PERMISSIVE:?}。\
             链不了 GPL 代码的输入法同样用不了 GPL 权重"
        ));
    }
    println!("许可 {license}");

    let attribution = model.attribution();
    if attribution.is_empty() {
        return fail("没有 attribution 字段。语料许可要求署名随权重传播");
    }
    if let Some(marker) = SHARE_ALIKE.iter().find(|m| attribution.contains(**m)) {
        return fail(&format!(
            "署名里出现 {marker}，该语料带 share-alike 义务，这样的权重不该发布。\
             署名原文: {attribution}"
        ));
    }
    println!("署名 {attribution}");

    // The engine's own first candidate is index 0 by convention, and the correct reading is first
    // here, so a model that works ranks it first and one that is broken does not.
    let scores = model.score("", &CHECK_SENTENCE);
    let best = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(usize::MAX);
    if best != 0 {
        return fail(&format!(
            "前向计算可疑: 正确读法「{}」被排在「{}」之后",
            CHECK_SENTENCE[0], CHECK_SENTENCE[best]
        ));
    }
    println!(
        "排序检查通过: 「{}」 {:.4}，次优 {:.4}",
        CHECK_SENTENCE[0],
        scores[0],
        scores[1..].iter().cloned().fold(f32::MIN, f32::max)
    );

    println!("通过");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::{sha256, PERMISSIVE, SHARE_ALIKE};

    /// The published vectors. A digest function that is wrong in a way the release notes happen to
    /// agree with would pass every check here while approving a file nobody else can verify, so it
    /// is checked against values this project did not produce.
    #[test]
    fn sha256_matches_the_published_vectors() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Longer than one 64-byte block, so the padding and the second round are exercised.
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn the_repositorys_own_licence_is_not_publishable() {
        // The bug this guards: export.py stamped every model with the repository's GPL-3.0, which
        // is share-alike and unusable by most of the input methods this exists for.
        assert!(!PERMISSIVE.contains(&"GPL-3.0-only"));
        assert!(!PERMISSIVE.contains(&"AGPL-3.0"));
        assert!(!PERMISSIVE.contains(&"CC-BY-SA-4.0"));
        assert!(PERMISSIVE.contains(&"Apache-2.0"));
    }

    #[test]
    fn share_alike_corpora_are_recognised_in_an_attribution() {
        let wiki = "Trained on Chinese Wikipedia (CC BY-SA 4.0, https://dumps.wikimedia.org/zhwiki/) and LCCC (MIT).";
        assert!(SHARE_ALIKE.iter().any(|m| wiki.contains(m)));
        let clean = "Trained on the Chinese portion of C4 (ODC-BY) and LCCC (MIT).";
        assert!(!SHARE_ALIKE.iter().any(|m| clean.contains(m)));
    }
}

/// FIPS 180-4, so that verifying a download needs nothing but this crate. Adding a dependency for
/// it would mean an adopter checking a model file has to trust one more package than the loader
/// itself does.
fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bits = (data.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bits.to_be_bytes());

    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}
