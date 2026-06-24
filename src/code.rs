//! Source-code gathering for the `code` surface.
//!
//! Lists source files under a project directory (by extension) and splits text
//! into bounded character chunks. The actual file *reads* are done by the caller
//! with a per-file timeout, because `~/Documents/dev` lives under iCloud and a
//! dataless/offloaded file would otherwise block `read_to_string` indefinitely.
//! Symlinks are not followed (avoids loops), and oversize/non-regular files are
//! skipped.

use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

const CODE_EXTS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "c", "h", "cpp", "cc", "hpp", "java", "rb",
    "scala", "swift", "kt", "ml", "hs", "jl", "lua", "sh", "sql", "ex", "exs", "clj", "zig",
];
// Skip non-representative trees. The point is "what does this project do", so
// archives, benchmarks, examples, tests, and docs are excluded — they otherwise
// dominate the alphabetically-first chunk budget and crowd out src/.
const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".cache",
    "vendor",
    "archive",
    "benches",
    "bench",
    "examples",
    "example",
    "tests",
    "test",
    "testdata",
    "fixtures",
    "docs",
    "doc",
    ".github",
];
const MAX_FILE_BYTES: u64 = 1_000_000;
const MAX_CHUNK_CHARS: usize = 2000;
/// Per-project chunk cap. Bounds cost; projects exceeding it report `capped`.
pub const MAX_CHUNKS: usize = 48;
/// Re-exported so the caller can size its chunking the same way.
pub const CHUNK_CHARS: usize = MAX_CHUNK_CHARS;

/// List source files under `dir` (recursive), sorted. Skips build/vendor dirs,
/// symlinks (no loop-following), and files larger than [`MAX_FILE_BYTES`].
pub fn list_source_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_files(dir, &mut files);
    // Order by representativeness so the chunk cap spends on API surface first:
    // entry points (lib/main/mod/__init__) -> anything under src/ -> the rest.
    files.sort_by(|a, b| priority(a).cmp(&priority(b)).then_with(|| a.cmp(b)));
    files
}

/// Lower = more representative of what the project is.
fn priority(p: &Path) -> u8 {
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if matches!(
        name,
        "lib.rs" | "main.rs" | "mod.rs" | "__init__.py" | "__main__.py"
    ) {
        0
    } else if p.components().any(|c| c.as_os_str() == "src") {
        1
    } else {
        2
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let Ok(ft) = entry.file_type() else { continue };
        let name = entry.file_name().to_string_lossy().into_owned();
        if ft.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) || name.starts_with('.') {
                continue;
            }
            collect_files(&entry.path(), out);
        } else if ft.is_file() {
            let path = entry.path();
            let ext_ok = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| CODE_EXTS.contains(&e));
            if !ext_ok {
                continue;
            }
            // Accept only resident files within the size cap. `blocks() == 0` with
            // a nonzero size marks an iCloud-offloaded (dataless) file; reading it
            // would block on a download, so skip it here without a read.
            if let Ok(m) = entry.metadata() {
                if m.blocks() > 0 && m.len() <= MAX_FILE_BYTES {
                    out.push(path);
                }
            }
        }
    }
}

/// Split text into chunks of at most `max` characters, on char boundaries.
pub fn chunk_text(text: &str, max: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = trimmed.chars().collect();
    chars.chunks(max).map(|c| c.iter().collect()).collect()
}
