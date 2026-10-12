//! Source-code gathering for the `code` surface.
//!
//! Lists source files under a project directory (by extension) and splits text
//! into bounded character chunks. The actual file *reads* are done by the caller
//! with a per-file timeout, because a cloud-synced project tree (e.g. iCloud)
//! can hold dataless/offloaded files that block `read_to_string` indefinitely.
//! Source symlinks are not followed (avoids loops), and oversize/non-regular files are
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

/// Controls project-local source selection. Hard exclusions and limits always apply.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct SourceOptions {
    /// Bypass project `.gitignore` and `.ignore` rules (default: respect them).
    pub include_ignored: bool,
}

/// List source files under `dir`, respecting project-local ignore rules.
/// Skips build/vendor dirs, source symlinks, and files over `MAX_FILE_BYTES`.
pub fn list_source_files(dir: &Path) -> Vec<PathBuf> {
    list_source_files_with_options(dir, SourceOptions::default())
}

/// List source files with explicit ignore selection. Parent/global ignore rules
/// and `.git/info/exclude` are not loaded. Local ignore files use standard ignore
/// semantics, including nested rules, negation, and linked ignore configuration.
pub fn list_source_files_with_options(dir: &Path, options: SourceOptions) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut walk = ignore::WalkBuilder::new(dir);
    walk.parents(false)
        .git_global(false)
        .git_exclude(false)
        .require_git(false)
        .git_ignore(!options.include_ignored)
        .ignore(!options.include_ignored)
        .hidden(false)
        .follow_links(false)
        .filter_entry(|entry| {
            if entry.depth() == 0 || !entry.file_type().is_some_and(|ft| ft.is_dir()) {
                return true;
            }
            let name = entry.file_name().to_string_lossy();
            !name.starts_with('.') && !SKIP_DIRS.contains(&name.as_ref())
        });
    for entry in walk.build().filter_map(Result::ok) {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        if !path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| CODE_EXTS.contains(&e))
        {
            continue;
        }
        // Keep resident-file and size limits independently of ignore selection.
        // A nonzero-size file with no blocks can be an offloaded iCloud file.
        if let Ok(m) = entry.metadata() {
            if m.blocks() > 0 && m.len() <= MAX_FILE_BYTES {
                files.push(path.to_path_buf());
            }
        }
    }
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

/// Split text into chunks of at most `max` characters, on char boundaries.
pub fn chunk_text(text: &str, max: usize) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = trimmed.chars().collect();
    chars.chunks(max).map(|c| c.iter().collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_ignore_rules_select_sources_with_nested_negation() {
        let root = std::env::temp_dir().join(format!(
            "scry-ignore-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let project = root.join("project");
        std::fs::create_dir_all(project.join("src/nested")).unwrap();
        // Parent rules must not affect this project, even without a .git directory.
        std::fs::write(root.join(".gitignore"), "*\n").unwrap();
        std::fs::write(root.join(".ignore"), "*\n").unwrap();
        std::fs::write(
            project.join(".gitignore"),
            "ignored.rs\nsrc/*.rs\n!src/keep.rs\n",
        )
        .unwrap();
        std::fs::write(project.join(".ignore"), "local.py\n").unwrap();
        std::fs::write(project.join("src/nested/.gitignore"), "*.rs\n!rescued.rs\n").unwrap();
        std::fs::write(project.join("src/nested/.ignore"), "*.py\n!rescued.py\n").unwrap();
        for name in [
            "lib.rs",
            "ignored.rs",
            "local.py",
            "src/drop.rs",
            "src/keep.rs",
            "src/nested/drop.rs",
            "src/nested/rescued.rs",
            "src/nested/drop.py",
            "src/nested/rescued.py",
        ] {
            std::fs::write(project.join(name), "synthetic source\n").unwrap();
        }
        let selected: Vec<_> = list_source_files(&project)
            .into_iter()
            .map(|p| p.strip_prefix(&project).unwrap().to_path_buf())
            .collect();
        assert_eq!(
            selected,
            [
                "lib.rs",
                "src/keep.rs",
                "src/nested/rescued.py",
                "src/nested/rescued.rs"
            ]
            .map(PathBuf::from)
        );
        // The override restores ignored sources, but never hard-excluded paths.
        for name in ["target/build.rs", "tests/test.rs", ".hidden/private.rs"] {
            let path = project.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "excluded synthetic source").unwrap();
        }
        std::os::unix::fs::symlink(project.join("lib.rs"), project.join("linked.rs")).unwrap();
        std::os::unix::fs::symlink(project.join("src"), project.join("linked-dir")).unwrap();
        std::fs::write(
            project.join("large.rs"),
            vec![b'x'; MAX_FILE_BYTES as usize + 1],
        )
        .unwrap();
        std::fs::write(project.join("not-code.txt"), "not source").unwrap();
        let options = SourceOptions {
            include_ignored: true,
        };
        let mut included: Vec<_> = list_source_files_with_options(&project, options)
            .into_iter()
            .map(|p| p.strip_prefix(&project).unwrap().to_path_buf())
            .collect();
        included.sort();
        assert_eq!(
            included,
            [
                "ignored.rs",
                "lib.rs",
                "local.py",
                "src/drop.rs",
                "src/keep.rs",
                "src/nested/drop.py",
                "src/nested/drop.rs",
                "src/nested/rescued.py",
                "src/nested/rescued.rs"
            ]
            .map(PathBuf::from)
        );
        assert_eq!(list_source_files(&project).len(), 4);
        std::fs::remove_dir_all(root).unwrap();
    }
}
