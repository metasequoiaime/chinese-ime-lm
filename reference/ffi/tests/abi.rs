//! Calls the exported functions the way a C host does and checks them against the Rust API they wrap: the same decisions, the same scores, and failures reported instead of crashing.

mod support;

use std::ffi::{c_char, CStr};
use std::ptr;
use std::sync::Arc;

use chinese_ime_lm::{CandidateFacts, Reranker, SentenceModel};
use chinese_ime_lm_ffi::*;
use support::fixture;

fn str_of(text: &str) -> CimelmStr {
    CimelmStr {
        data: text.as_ptr().cast::<c_char>(),
        len: text.len(),
    }
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(cimelm_last_error()) }
        .to_str()
        .expect("error messages are UTF-8")
        .to_owned()
}

fn load() -> *mut CimelmModel {
    let bytes = fixture::model_bytes();
    let mut model = ptr::null_mut();
    let status = unsafe { cimelm_model_load(bytes.as_ptr(), bytes.len(), &mut model) };
    assert_eq!(status, CimelmStatus::Ok, "{}", last_error());
    assert!(!model.is_null());
    model
}

fn reranker(model: *const CimelmModel) -> *mut CimelmReranker {
    let mut reranker = ptr::null_mut();
    let status = unsafe { cimelm_reranker_new(model, &mut reranker) };
    assert_eq!(status, CimelmStatus::Ok, "{}", last_error());
    reranker
}

fn best_where(
    reranker: *mut CimelmReranker,
    context: &str,
    texts: &[&str],
    answers_key: &[u8],
    trusted: &[u8],
) -> (CimelmStatus, usize) {
    let candidates: Vec<CimelmStr> = texts.iter().map(|text| str_of(text)).collect();
    let mut index = usize::MAX;
    let status = unsafe {
        cimelm_reranker_best_where(
            reranker,
            str_of(context),
            candidates.as_ptr(),
            candidates.len(),
            answers_key.as_ptr(),
            trusted.as_ptr(),
            &mut index,
        )
    };
    (status, index)
}

#[test]
fn version_is_the_crate_version() {
    let version = unsafe { CStr::from_ptr(cimelm_version()) };
    assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn model_exposes_its_licence_and_attribution() {
    let model = load();
    let license = unsafe { CStr::from_ptr(cimelm_model_license(model)) };
    let attribution = unsafe { CStr::from_ptr(cimelm_model_attribution(model)) };
    assert_eq!(license.to_str().unwrap(), fixture::LICENSE);
    assert_eq!(attribution.to_str().unwrap(), fixture::ATTRIBUTION);
    assert!(unsafe { cimelm_model_license(ptr::null()) }.is_null());
    assert!(unsafe { cimelm_model_attribution(ptr::null()) }.is_null());
    unsafe { cimelm_model_free(model) };
}

#[test]
fn decisions_and_scores_match_the_rust_api() {
    let model = load();
    let handle = reranker(model);
    let mut rust = Reranker::new(Arc::new(
        SentenceModel::load(&fixture::model_bytes()).expect("load"),
    ));
    let facts = |_| CandidateFacts {
        answers_key: true,
        trusted_dictionary_hit: false,
    };

    // Whichever of the two the model prefers, it is first in one order and second in the other, so one of these is a promotion and the other is not.
    let mut promoted = 0;
    for texts in [["你好", "我很"], ["我很", "你好"]] {
        let (status, index) = best_where(handle, "你好吗", &texts, &[1, 1], &[0, 0]);
        assert_eq!(status, CimelmStatus::Ok, "{}", last_error());
        assert_eq!(index, rust.best_where("你好吗", &texts, facts).unwrap_or(0));
        promoted += usize::from(index != 0);

        let candidates: Vec<CimelmStr> = texts.iter().map(|text| str_of(text)).collect();
        let mut scores = [0.0f32; 2];
        let status = unsafe {
            cimelm_reranker_log_probabilities(
                handle,
                str_of("你好吗"),
                candidates.as_ptr(),
                candidates.len(),
                scores.as_mut_ptr(),
            )
        };
        assert_eq!(status, CimelmStatus::Ok, "{}", last_error());
        let expected = rust.log_probabilities("你好吗", &texts);
        assert_eq!(
            scores.map(f32::to_bits).to_vec(),
            expected
                .iter()
                .map(|score| score.to_bits())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(promoted, 1, "exactly one of the two orders is a promotion");

    unsafe {
        cimelm_reranker_free(handle);
        cimelm_model_free(model);
    }
}

#[test]
fn gating_leaves_a_trusted_leader_alone() {
    let model = load();
    let handle = reranker(model);
    for texts in [["你好", "我很"], ["我很", "你好"]] {
        let (status, index) = best_where(handle, "", &texts, &[1, 1], &[1, 0]);
        assert_eq!(status, CimelmStatus::Ok);
        assert_eq!(
            index, 0,
            "a trusted dictionary hit in first place is never overridden"
        );
        let (status, index) = best_where(handle, "", &texts, &[1, 0], &[0, 0]);
        assert_eq!(status, CimelmStatus::Ok);
        assert_eq!(
            index, 0,
            "a candidate that does not answer the key is not comparable"
        );
    }
    unsafe {
        cimelm_reranker_free(handle);
        cimelm_model_free(model);
    }
}

#[test]
fn an_empty_candidate_list_needs_no_arrays() {
    let model = load();
    let handle = reranker(model);
    let mut index = usize::MAX;
    let status = unsafe {
        cimelm_reranker_best_where(
            handle,
            str_of(""),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            &mut index,
        )
    };
    assert_eq!(status, CimelmStatus::Ok, "{}", last_error());
    assert_eq!(index, 0);
    let status = unsafe {
        cimelm_reranker_log_probabilities(handle, str_of(""), ptr::null(), 0, ptr::null_mut())
    };
    assert_eq!(status, CimelmStatus::Ok, "{}", last_error());
    unsafe {
        cimelm_reranker_free(handle);
        cimelm_model_free(model);
    }
}

#[test]
fn a_reranker_outlives_its_model_handle() {
    let model = load();
    let handle = reranker(model);
    unsafe { cimelm_model_free(model) };
    let (status, _) = best_where(handle, "你", &["你好", "我很"], &[1, 1], &[0, 0]);
    assert_eq!(status, CimelmStatus::Ok, "{}", last_error());
    unsafe { cimelm_reranker_free(handle) };
}

#[test]
fn malformed_model_bytes_are_rejected() {
    let mut model = ptr::NonNull::<CimelmModel>::dangling().as_ptr();
    let garbage = [0xffu8; 32];
    let status = unsafe { cimelm_model_load(garbage.as_ptr(), garbage.len(), &mut model) };
    assert_eq!(status, CimelmStatus::InvalidModel);
    assert!(model.is_null(), "a failed load leaves no dangling handle");
    assert!(!last_error().is_empty());

    let bytes = fixture::model_bytes();
    let truncated = &bytes[..bytes.len() / 2];
    let status = unsafe { cimelm_model_load(truncated.as_ptr(), truncated.len(), &mut model) };
    assert_eq!(status, CimelmStatus::InvalidModel);
    assert!(model.is_null());
}

#[test]
fn bad_pointers_and_strings_are_reported() {
    let mut model = ptr::null_mut();
    let status = unsafe { cimelm_model_load(ptr::null(), 10, &mut model) };
    assert_eq!(status, CimelmStatus::InvalidArgument);
    assert!(last_error().contains("bytes"), "{}", last_error());

    let bytes = fixture::model_bytes();
    let status = unsafe { cimelm_model_load(bytes.as_ptr(), bytes.len(), ptr::null_mut()) };
    assert_eq!(status, CimelmStatus::InvalidArgument);

    let mut handle = ptr::null_mut();
    let status = unsafe { cimelm_reranker_new(ptr::null(), &mut handle) };
    assert_eq!(status, CimelmStatus::InvalidArgument);
    assert!(handle.is_null());

    let (status, _) = best_where(ptr::null_mut(), "", &["你"], &[1], &[0]);
    assert_eq!(status, CimelmStatus::InvalidArgument);

    let model = load();
    let handle = reranker(model);
    let invalid = CimelmStr {
        data: b"\xff\xfe".as_ptr().cast(),
        len: 2,
    };
    let candidates = [str_of("你好"), invalid];
    let mut index = 0;
    let status = unsafe {
        cimelm_reranker_best_where(
            handle,
            str_of(""),
            candidates.as_ptr(),
            2,
            [1u8, 1].as_ptr(),
            [0u8, 0].as_ptr(),
            &mut index,
        )
    };
    assert_eq!(status, CimelmStatus::InvalidArgument);
    assert!(last_error().contains("candidates[1]"), "{}", last_error());

    let status = unsafe {
        cimelm_reranker_log_probabilities(
            handle,
            str_of(""),
            candidates.as_ptr(),
            1,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, CimelmStatus::InvalidArgument);
    assert!(last_error().contains("out_scores"), "{}", last_error());

    // A rejected call leaves the reranker usable.
    let (status, _) = best_where(handle, "", &["你好", "我很"], &[1, 1], &[0, 0]);
    assert_eq!(status, CimelmStatus::Ok, "{}", last_error());

    unsafe {
        cimelm_reranker_free(handle);
        cimelm_model_free(model);
        cimelm_reranker_free(ptr::null_mut());
        cimelm_model_free(ptr::null_mut());
    }
}
