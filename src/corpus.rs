//! Project discovery and scope lenses.

use std::path::Path;

/// One project: its directory name, path, and (truncated) README text.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub name: String,
    pub dir: std::path::PathBuf,
    pub readme: String,
}

const CHAR_BUDGET: usize = 24_000;
const SKIP_NAMES: &[&str] = &["target", "node_modules", "scry"];

/// Named scope shorthands: `(name, instruction)`. The instruction is prepended
/// to text in Qwen3-Embedding's `Instruct: {task}\nQuery: {text}` form.
pub const DEFAULT_SCOPES: &[(&str, &str)] = &[
    ("purpose", "Represent this software project by its core purpose and the problem it solves."),
    ("techniques", "Represent this software project by the algorithms, methods, and technical techniques it uses."),
    ("domain", "Represent this software project by its subject domain and the field it belongs to."),
];

/// Discover projects: immediate subdirectories of `root` that contain a
/// `README*` file, skipping dotted/underscored names and build dirs.
pub fn discover(root: &Path) -> std::io::Result<Vec<Project>> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if SKIP_NAMES.contains(&name.as_str()) || name.starts_with('_') || name.starts_with('.') {
            continue;
        }
        let Some(readme) = find_readme(&entry.path()) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&readme) else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        // Lead the embedded text with dense metadata (Cargo/package description
        // + keywords): the highest-signal "what is this" summary, otherwise
        // missing from many short READMEs.
        let meta = read_meta(&entry.path());
        let body = if meta.is_empty() {
            text.to_string()
        } else {
            format!("{meta}\n\n{text}")
        };
        let truncated: String = body.chars().take(CHAR_BUDGET).collect();
        out.push(Project {
            name,
            dir: entry.path(),
            readme: truncated,
        });
    }
    Ok(out)
}

fn find_readme(dir: &Path) -> Option<std::path::PathBuf> {
    let mut hits: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("README"))
        })
        .collect();
    hits.sort();
    hits.into_iter().next()
}

/// Format one input in the instruction-conditioned form.
pub fn format_input(instruction: &str, text: &str) -> String {
    format!("Instruct: {instruction}\nQuery: {text}")
}

/// Dense one-line project metadata: Cargo.toml description + keywords, falling
/// back to package.json description. Best-effort line parsing (no toml/json dep).
fn read_meta(dir: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Ok(t) = std::fs::read_to_string(dir.join("Cargo.toml")) {
        if let Some(d) = toml_value(&t, "description") {
            parts.push(d);
        }
        if let Some(k) = toml_array(&t, "keywords") {
            parts.push(format!("keywords: {k}"));
        }
    }
    if parts.is_empty() {
        if let Ok(t) = std::fs::read_to_string(dir.join("package.json")) {
            if let Some(d) = json_value(&t, "description") {
                parts.push(d);
            }
        }
    }
    parts.join(". ")
}

/// First `key = "..."` value at the start of a line (Cargo.toml scalar).
fn toml_value(t: &str, key: &str) -> Option<String> {
    for line in t.lines() {
        let l = line.trim_start();
        if let Some(after) = l.strip_prefix(key) {
            let after = after.trim_start();
            if let Some(rhs) = after.strip_prefix('=') {
                if let Some(inner) = rhs.trim().strip_prefix('"') {
                    if let Some(end) = inner.find('"') {
                        return Some(inner[..end].to_string());
                    }
                }
            }
        }
    }
    None
}

/// First single-line `key = ["a", "b"]` array, joined as "a, b".
fn toml_array(t: &str, key: &str) -> Option<String> {
    for line in t.lines() {
        let l = line.trim_start();
        if let Some(after) = l.strip_prefix(key) {
            let after = after.trim_start();
            if let Some(rhs) = after.strip_prefix('=') {
                if let Some(inner) = rhs.trim().strip_prefix('[') {
                    let end = inner.find(']').unwrap_or(inner.len());
                    let items: Vec<String> = inner[..end]
                        .split(',')
                        .map(|s| s.trim().trim_matches('"').trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if !items.is_empty() {
                        return Some(items.join(", "));
                    }
                }
            }
        }
    }
    None
}

/// `"key": "value"` from a JSON blob (best-effort, no escape handling).
fn json_value(t: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = t.find(&pat)?;
    let rhs = t[i + pat.len()..]
        .trim_start()
        .strip_prefix(':')?
        .trim_start();
    let inner = rhs.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

/// Resolve a named scope to its instruction, or pass free text through.
pub fn resolve_scope(name_or_instruction: &str) -> String {
    DEFAULT_SCOPES
        .iter()
        .find(|(n, _)| *n == name_or_instruction)
        .map(|(_, instr)| instr.to_string())
        .unwrap_or_else(|| name_or_instruction.to_string())
}

/// The default project root: `~/Documents/dev`.
pub fn default_root() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    Path::new(&home).join("Documents/dev")
}
