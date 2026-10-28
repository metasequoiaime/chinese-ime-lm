//! Writes the test model to a file, so that the C smoke test in CI loads exactly the bytes the Rust tests do.
//!
//! usage: write_fixture <path>

#[path = "../tests/support/fixture.rs"]
mod fixture;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: write_fixture <path>");
    std::fs::write(&path, fixture::model_bytes()).expect("write the model file");
}
