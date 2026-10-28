//! C ABI for [`chinese_ime_lm`], so that input methods written in C, C++, Swift, Kotlin or anything else with a C FFI can rerank candidates with the same model and the same gating as a Rust engine.
//!
//! The header is `include/chinese_ime_lm.h`, generated from this file by cbindgen. Every exported function catches panics, checks its pointers and reports failure through [`CimelmStatus`] and [`cimelm_last_error`]; none of them unwinds into the caller.
//!
//! Only [`Reranker::best_where`](chinese_ime_lm::Reranker::best_where) is exported, not `Reranker::best`: `best` infers its two facts from one engine's source numbering, and a C caller is never that engine.

use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;

use chinese_ime_lm::{CandidateFacts, Reranker, SentenceModel};

/// The result of every fallible call. On anything but `CIMELM_STATUS_OK`, `cimelm_last_error` describes what went wrong.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CimelmStatus {
    /// The call succeeded and wrote its outputs.
    Ok = 0,
    /// A pointer was null or misaligned, a length was too large, or a string was not UTF-8. Nothing was written.
    InvalidArgument = 1,
    /// The bytes passed to `cimelm_model_load` are not a model this library can load.
    InvalidModel = 2,
    /// A bug inside the library panicked. The panic was caught; a reranker it happened in refuses every later call and has to be freed.
    Panic = 3,
}

/// A UTF-8 string that is not NUL-terminated.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CimelmStr {
    /// The bytes. May be null when `len` is 0.
    pub data: *const c_char,
    /// The length in bytes, not in characters.
    pub len: usize,
}

/// A loaded model. Read-only, so one model may be shared by any number of rerankers on any number of threads.
pub struct CimelmModel {
    model: Arc<SentenceModel>,
    license: CString,
    attribution: CString,
}

/// The per-session state of reranking: the model plus the committed text it last ran. Create one per input session. It may move between threads, but must not be used from two threads at once.
pub struct CimelmReranker {
    reranker: Reranker,
    /// Set for the duration of every call and cleared when the call returns. Still set at the start of a call means a panic interrupted an earlier one and the cached state may be half written.
    poisoned: bool,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Records `message` as this thread's last error and returns `status`, so a failure can be reported in one expression.
fn fail(status: CimelmStatus, message: impl Into<String>) -> CimelmStatus {
    // The message ends up behind a `const char *`, so an interior NUL would silently truncate it.
    let message = message.into().replace('\0', " ");
    let message = CString::new(message).unwrap_or_default();
    let _ = LAST_ERROR.try_with(|slot| *slot.borrow_mut() = message);
    status
}

/// Runs the body of an exported function, turning a panic into `CIMELM_STATUS_PANIC` instead of letting it cross the C boundary.
fn guard(body: impl FnOnce() -> Result<CimelmStatus, CimelmStatus>) -> CimelmStatus {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(status) | Err(status)) => status,
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("no message");
            fail(CimelmStatus::Panic, format!("internal panic: {detail}"))
        }
    }
}

/// Borrows `len` elements starting at `data`.
///
/// # Safety
///
/// When `len` is not 0, `data` must point to `len` initialized elements that stay valid and unmodified for the rest of the call.
unsafe fn array<'a, T>(data: *const T, len: usize, what: &str) -> Result<&'a [T], CimelmStatus> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(fail(
            CimelmStatus::InvalidArgument,
            format!("{what} is null but its length is {len}"),
        ));
    }
    if data as usize % align_of::<T>() != 0 {
        return Err(fail(
            CimelmStatus::InvalidArgument,
            format!("{what} is misaligned"),
        ));
    }
    if len
        .checked_mul(size_of::<T>())
        .map_or(true, |bytes| bytes > isize::MAX as usize)
    {
        return Err(fail(
            CimelmStatus::InvalidArgument,
            format!("{what} length {len} is too large"),
        ));
    }
    // SAFETY: non-null, aligned and within isize::MAX bytes were checked above; that the memory is readable and initialized is the caller's promise.
    Ok(unsafe { std::slice::from_raw_parts(data, len) })
}

/// Borrows the single element `target` points to, mutably.
///
/// # Safety
///
/// `target` must be null or point to a valid, writable `T` that nothing else accesses for the rest of the call.
unsafe fn place<'a, T>(target: *mut T, what: &str) -> Result<&'a mut T, CimelmStatus> {
    if target.is_null() {
        return Err(fail(
            CimelmStatus::InvalidArgument,
            format!("{what} is null"),
        ));
    }
    if target as usize % align_of::<T>() != 0 {
        return Err(fail(
            CimelmStatus::InvalidArgument,
            format!("{what} is misaligned"),
        ));
    }
    // SAFETY: non-null and aligned were checked above; validity and exclusive access are the caller's promise.
    Ok(unsafe { &mut *target })
}

/// Borrows a `CimelmStr` as `&str`.
///
/// # Safety
///
/// As for [`array`], with `value.data` and `value.len`.
unsafe fn text<'a>(value: CimelmStr, what: &str) -> Result<&'a str, CimelmStatus> {
    // SAFETY: forwarded from this function's own contract.
    let bytes = unsafe { array(value.data.cast::<u8>(), value.len, what)? };
    std::str::from_utf8(bytes).map_err(|_| {
        fail(
            CimelmStatus::InvalidArgument,
            format!("{what} is not valid UTF-8"),
        )
    })
}

/// Borrows `count` candidate strings.
///
/// # Safety
///
/// As for [`array`] with `candidates` and `count`, and as for [`text`] with every element.
unsafe fn candidate_texts<'a>(
    candidates: *const CimelmStr,
    count: usize,
) -> Result<Vec<&'a str>, CimelmStatus> {
    // SAFETY: forwarded from this function's own contract.
    let candidates = unsafe { array(candidates, count, "candidates")? };
    candidates
        .iter()
        .enumerate()
        // SAFETY: forwarded from this function's own contract.
        .map(|(index, value)| unsafe { text(*value, &format!("candidates[{index}]")) })
        .collect()
}

/// Runs `body` on a reranker, refusing one that an earlier panic left half written.
///
/// # Safety
///
/// As for [`place`] with `reranker`.
unsafe fn with_reranker(
    reranker: *mut CimelmReranker,
    body: impl FnOnce(&mut Reranker) -> Result<CimelmStatus, CimelmStatus>,
) -> CimelmStatus {
    guard(|| {
        // SAFETY: forwarded from this function's own contract.
        let handle = unsafe { place(reranker, "reranker")? };
        if handle.poisoned {
            return Err(fail(
                CimelmStatus::Panic,
                "this reranker was interrupted by an earlier panic; free it and create a new one",
            ));
        }
        handle.poisoned = true;
        let status = body(&mut handle.reranker);
        handle.poisoned = false;
        status
    })
}

/// The library version, such as `"0.1.0"`. A static string; do not free it.
#[no_mangle]
pub extern "C" fn cimelm_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// What went wrong in the most recent call on this thread that did not return `CIMELM_STATUS_OK`, as a NUL-terminated UTF-8 string. Owned by the library and valid until the next failing call on the same thread. Calls that succeed leave it unchanged, so read it only after a failure.
#[no_mangle]
pub extern "C" fn cimelm_last_error() -> *const c_char {
    LAST_ERROR
        .try_with(|slot| slot.borrow().as_ptr())
        .unwrap_or(b"\0".as_ptr().cast())
}

/// Loads a model from the bytes of a `sentence-model.safetensors` file. The bytes are only read during the call, so the caller may free them as soon as it returns.
///
/// On success `*out` receives a model to release with `cimelm_model_free`. On failure `*out` is set to null. A model file is untrusted input: a malformed one is answered with `CIMELM_STATUS_INVALID_MODEL`, never by reading out of bounds.
///
/// # Safety
///
/// `bytes` must point to `len` readable bytes (it may be null when `len` is 0), and `out` must point to a writable `cimelm_model *`.
#[no_mangle]
pub unsafe extern "C" fn cimelm_model_load(
    bytes: *const u8,
    len: usize,
    out: *mut *mut CimelmModel,
) -> CimelmStatus {
    guard(|| {
        // SAFETY: the caller promises `out` is writable.
        let out = unsafe { place(out, "out")? };
        *out = ptr::null_mut();
        // SAFETY: the caller promises `bytes` covers `len` readable bytes.
        let bytes = unsafe { array(bytes, len, "bytes")? };
        let model = SentenceModel::load(bytes)
            .map_err(|error| fail(CimelmStatus::InvalidModel, error.to_string()))?;
        let license = CString::new(model.license()).map_err(|_| {
            fail(
                CimelmStatus::InvalidModel,
                "the licence metadata contains a NUL byte",
            )
        })?;
        let attribution = CString::new(model.attribution()).map_err(|_| {
            fail(
                CimelmStatus::InvalidModel,
                "the attribution metadata contains a NUL byte",
            )
        })?;
        *out = Box::into_raw(Box::new(CimelmModel {
            model: Arc::new(model),
            license,
            attribution,
        }));
        Ok(CimelmStatus::Ok)
    })
}

/// Releases a model. Rerankers created from it keep working: each holds its own reference to the weights. Null is ignored.
///
/// # Safety
///
/// `model` must be null or a pointer returned by `cimelm_model_load` that has not been freed, and no other thread may be using it.
#[no_mangle]
pub unsafe extern "C" fn cimelm_model_free(model: *mut CimelmModel) {
    if !model.is_null() {
        // SAFETY: the caller promises this pointer came from Box::into_raw in cimelm_model_load and is freed once.
        drop(unsafe { Box::from_raw(model) });
    }
}

/// The licence of the weights as an SPDX identifier, such as `"Apache-2.0"`; this is not the licence of the code that produced them. Empty when the file does not declare one, which older files do not: treat that as unknown, not as permission. Owned by the model and valid until it is freed. Null when `model` is null.
///
/// # Safety
///
/// `model` must be null or a live pointer returned by `cimelm_model_load`.
#[no_mangle]
pub unsafe extern "C" fn cimelm_model_license(model: *const CimelmModel) -> *const c_char {
    // SAFETY: the caller promises a non-null pointer is a live model.
    match unsafe { model.as_ref() } {
        Some(model) => model.license.as_ptr(),
        None => ptr::null(),
    }
}

/// The corpus attribution stored in the weights file. The corpora's licences require it to travel with the model, so keep it when you redistribute one. Owned by the model and valid until it is freed. Null when `model` is null.
///
/// # Safety
///
/// `model` must be null or a live pointer returned by `cimelm_model_load`.
#[no_mangle]
pub unsafe extern "C" fn cimelm_model_attribution(model: *const CimelmModel) -> *const c_char {
    // SAFETY: the caller promises a non-null pointer is a live model.
    match unsafe { model.as_ref() } {
        Some(model) => model.attribution.as_ptr(),
        None => ptr::null(),
    }
}

/// Creates a reranker for one input session, typically one per focus change. It caches the committed text it last ran, which is what keeps a keystroke inside a frame, so do not share one between sessions.
///
/// On success `*out` receives a reranker to release with `cimelm_reranker_free`. On failure `*out` is set to null.
///
/// # Safety
///
/// `model` must be a live pointer returned by `cimelm_model_load`, and `out` must point to a writable `cimelm_reranker *`.
#[no_mangle]
pub unsafe extern "C" fn cimelm_reranker_new(
    model: *const CimelmModel,
    out: *mut *mut CimelmReranker,
) -> CimelmStatus {
    guard(|| {
        // SAFETY: the caller promises `out` is writable.
        let out = unsafe { place(out, "out")? };
        *out = ptr::null_mut();
        // SAFETY: the caller promises a non-null pointer is a live model.
        let model = unsafe { model.as_ref() }
            .ok_or_else(|| fail(CimelmStatus::InvalidArgument, "model is null"))?;
        *out = Box::into_raw(Box::new(CimelmReranker {
            reranker: Reranker::new(Arc::clone(&model.model)),
            poisoned: false,
        }));
        Ok(CimelmStatus::Ok)
    })
}

/// Releases a reranker. Null is ignored.
///
/// # Safety
///
/// `reranker` must be null or a pointer returned by `cimelm_reranker_new` that has not been freed, and no other thread may be using it.
#[no_mangle]
pub unsafe extern "C" fn cimelm_reranker_free(reranker: *mut CimelmReranker) {
    if !reranker.is_null() {
        // SAFETY: the caller promises this pointer came from Box::into_raw in cimelm_reranker_new and is freed once.
        drop(unsafe { Box::from_raw(reranker) });
    }
}

/// Decides which candidate the model would put first. `candidates` is the engine's list in the engine's own order, and the two flag arrays describe it index by index; a flag is true when non-zero.
///
/// On success `*out_index` is the index of the candidate to show first. 0 means keep the engine's order: the engine's leader stays first, either because the model agrees or because reranking was declined. Reranking is declined when the leader is a trusted dictionary hit, when the leader does not answer the key, or when fewer than two candidates answer it. At most nine candidates that answer the key are compared.
///
/// `answers_key[i]`: candidate `i` answers the whole key, rather than a prefix of it or a predictive completion running past it. Candidates that answer the key are compared even when they have different numbers of characters, because `xian` reads as 现 or as 西安 and both consume the key; prefixes and completions answer a different key. Character count is not a substitute for this flag: it agrees while a key has one segmentation and disagrees exactly where correction matters.
///
/// `trusted_dictionary_hit[i]`: candidate `i` is an exact dictionary hit that the engine vouches for as evidence of what the user meant. Overriding such a hit loses accuracy, because a dictionary that answered the whole key carries corpus frequency the model does not have. An engine that corrects input must report false for hits on the uncorrected reading: when the user types `gongsi` meaning `gongshi`, 公司 is an exact hit on the letters that arrived and evidence for a word nobody asked for.
///
/// `context` is the committed text before the cursor; only its tail that fits the model's context window is used.
///
/// # Safety
///
/// `reranker` must be a live pointer returned by `cimelm_reranker_new` that no other thread is using. `context` must describe readable bytes. `candidates`, `answers_key` and `trusted_dictionary_hit` must each point to `count` readable elements (they may be null when `count` is 0), every candidate must describe readable bytes, and `out_index` must point to a writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn cimelm_reranker_best_where(
    reranker: *mut CimelmReranker,
    context: CimelmStr,
    candidates: *const CimelmStr,
    count: usize,
    answers_key: *const u8,
    trusted_dictionary_hit: *const u8,
    out_index: *mut usize,
) -> CimelmStatus {
    // SAFETY: the caller promises `reranker` is live and used by this thread alone.
    unsafe {
        with_reranker(reranker, |reranker| {
            // SAFETY: every pointer below is covered by the caller's promises, as documented above.
            let out_index = place(out_index, "out_index")?;
            let context = text(context, "context")?;
            let texts = candidate_texts(candidates, count)?;
            let answers_key = array(answers_key, count, "answers_key")?;
            let trusted = array(trusted_dictionary_hit, count, "trusted_dictionary_hit")?;
            let promoted = reranker.best_where(context, &texts, |index| CandidateFacts {
                answers_key: answers_key[index] != 0,
                trusted_dictionary_hit: trusted[index] != 0,
            });
            *out_index = promoted.unwrap_or(0);
            Ok(CimelmStatus::Ok)
        })
    }
}

/// The model's mean log-probability per character for each candidate, in the order given, written to `out_scores[0..count]`. A candidate with no characters scores negative infinity.
///
/// These are the numbers behind `cimelm_reranker_best_where`, without its gating. They are for combining the model's opinion with evidence the engine already has; to let the model decide, call `cimelm_reranker_best_where`, which applies the gating the measurements require.
///
/// # Safety
///
/// `reranker` must be a live pointer returned by `cimelm_reranker_new` that no other thread is using. `context` must describe readable bytes. `candidates` must point to `count` readable elements and `out_scores` to `count` writable floats (both may be null when `count` is 0), and every candidate must describe readable bytes.
#[no_mangle]
pub unsafe extern "C" fn cimelm_reranker_log_probabilities(
    reranker: *mut CimelmReranker,
    context: CimelmStr,
    candidates: *const CimelmStr,
    count: usize,
    out_scores: *mut f32,
) -> CimelmStatus {
    // SAFETY: the caller promises `reranker` is live and used by this thread alone.
    unsafe {
        with_reranker(reranker, |reranker| {
            // SAFETY: every pointer below is covered by the caller's promises, as documented above.
            let context = text(context, "context")?;
            let texts = candidate_texts(candidates, count)?;
            // Validated before scoring so that a bad output pointer costs nothing and leaves the reranker's cache as it was.
            let out_scores: &mut [f32] = if count == 0 {
                &mut []
            } else {
                array(out_scores.cast_const(), count, "out_scores")?;
                std::slice::from_raw_parts_mut(out_scores, count)
            };
            let scores = reranker.log_probabilities(context, &texts);
            out_scores.copy_from_slice(&scores);
            Ok(CimelmStatus::Ok)
        })
    }
}

#[cfg(test)]
#[path = "../tests/support/fixture.rs"]
mod fixture;

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    fn message() -> String {
        // SAFETY: cimelm_last_error always returns a NUL-terminated string owned by this thread.
        unsafe { CStr::from_ptr(cimelm_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn a_panic_is_caught_and_poisons_the_reranker() {
        let model = Arc::new(SentenceModel::load(&fixture::model_bytes()).expect("load"));
        let mut handle = CimelmReranker {
            reranker: Reranker::new(model),
            poisoned: false,
        };

        // SAFETY: `handle` is a live, exclusively borrowed reranker.
        let status = unsafe { with_reranker(&mut handle, |_| panic!("deliberate")) };
        assert_eq!(status, CimelmStatus::Panic);
        assert!(message().contains("deliberate"), "{}", message());
        assert!(handle.poisoned);

        // The cache may be half written, so every later call is refused rather than trusted.
        // SAFETY: as above.
        let status = unsafe { with_reranker(&mut handle, |_| Ok(CimelmStatus::Ok)) };
        assert_eq!(status, CimelmStatus::Panic);
        assert!(message().contains("free it"), "{}", message());
    }

    #[test]
    fn an_ordinary_failure_does_not_poison() {
        let model = Arc::new(SentenceModel::load(&fixture::model_bytes()).expect("load"));
        let mut handle = CimelmReranker {
            reranker: Reranker::new(model),
            poisoned: false,
        };
        // SAFETY: `handle` is a live, exclusively borrowed reranker.
        let status = unsafe {
            with_reranker(&mut handle, |_| {
                Err(fail(CimelmStatus::InvalidArgument, "rejected"))
            })
        };
        assert_eq!(status, CimelmStatus::InvalidArgument);
        assert!(!handle.poisoned);
    }

    #[test]
    fn interior_nul_does_not_truncate_the_message() {
        fail(CimelmStatus::InvalidArgument, "before\0after");
        assert_eq!(message(), "before after");
    }
}
