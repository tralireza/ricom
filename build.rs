//! Build script: stamp the git short hash + commit date (YYMMDD) into the binary
//! so `--version` and the startup log identify the exact build.
//!
//! Resolution order, independently for the hash and the date:
//!   1. env override — `RICOM_GIT_HASH` / `RICOM_GIT_DATE`. The deploy step sets
//!      these because the build host's own `.git` may not match the synced
//!      sources (this Mac is the source of truth; i7 is scp-fed, not pushed).
//!   2. the `git` CLI in this checkout.
//!   3. a safe fallback ("unknown" hash, empty date) so a build never fails for
//!      lack of git (e.g. a source tarball).

use std::path::Path;
use std::process::Command;

fn main() {
    let hash = env_override("RICOM_GIT_HASH")
        .or_else(git_hash)
        .unwrap_or_else(|| "unknown".into());
    let date = env_override("RICOM_GIT_DATE")
        .or_else(git_date)
        .unwrap_or_default();

    println!("cargo:rustc-env=RICOM_GIT_HASH={hash}");
    println!("cargo:rustc-env=RICOM_GIT_DATE={date}");

    // Refresh the stamp when the override changes (deploy path) or when the
    // commit / working tree changes (plain-checkout path).
    println!("cargo:rerun-if-env-changed=RICOM_GIT_HASH");
    println!("cargo:rerun-if-env-changed=RICOM_GIT_DATE");
    for p in [".git/HEAD", ".git/index"] {
        if Path::new(p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    }
}

/// A non-empty environment variable, else `None`.
fn env_override(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

/// `git rev-parse --short HEAD`, with a `-dirty` suffix when *tracked* files
/// differ from HEAD. Untracked scratch (screenshots, plan files) is ignored.
fn git_hash() -> Option<String> {
    let hash = git(&["rev-parse", "--short", "HEAD"])?;
    // diff-index exit code: 0 = clean, 1 = tracked changes present.
    let dirty = Command::new("git")
        .args(["diff-index", "--quiet", "HEAD", "--"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(false);
    Some(if dirty { format!("{hash}-dirty") } else { hash })
}

/// The HEAD commit date formatted as YYMMDD.
fn git_date() -> Option<String> {
    git(&["show", "-s", "--format=%cd", "--date=format:%y%m%d", "HEAD"])
}

/// Run `git <args>` and return trimmed stdout, or `None` on any failure.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}
