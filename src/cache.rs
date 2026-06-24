//! Disk-backed embedding cache.
//!
//! Keyed by SHA-256 of `(model, provider, dimensions, exact input text)`. The
//! scope instruction is part of the input text, so changing the lens correctly
//! produces new embeddings while an unchanged input is never re-embedded.
//!
//! Interior mutability (a `Mutex` over the map, atomic counters) lets callers
//! share `&Cache` across concurrent embedding flows rather than threading
//! `&mut`, so batches of `query`/`ask` can run in parallel. Locks are held only
//! around in-memory get/insert, never across a network await.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Concurrency-safe embedding cache plus hit/miss counters for the run.
pub struct Cache {
    path: PathBuf,
    map: Mutex<HashMap<String, Vec<f32>>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl Cache {
    /// Load the cache from `path` (empty if it does not exist or is corrupt).
    pub fn load(path: PathBuf) -> Self {
        let map: HashMap<String, Vec<f32>> = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        Self {
            path,
            map: Mutex::new(map),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Cache key for one input. `provider` and `dimensions` are part of the key
    /// so different upstream providers and truncated/full-width embeddings never
    /// collide.
    pub fn key(model: &str, provider: Option<&str>, dimensions: Option<u32>, text: &str) -> String {
        let mut h = Sha256::new();
        h.update(model.as_bytes());
        h.update([0]);
        h.update(provider.unwrap_or("").as_bytes());
        h.update([0]);
        h.update(format!("{dimensions:?}").as_bytes());
        h.update([0]);
        h.update(text.as_bytes());
        format!("{:x}", h.finalize())
    }

    /// Presence flag per key (one lock acquisition).
    pub fn contains_each(&self, keys: &[String]) -> Vec<bool> {
        let m = self.map.lock().unwrap();
        keys.iter().map(|k| m.contains_key(k)).collect()
    }

    /// Insert one embedding.
    pub fn insert(&self, key: String, vec: Vec<f32>) {
        self.map.lock().unwrap().insert(key, vec);
    }

    /// Clone the vectors for `keys` (each must be present; missing -> empty).
    pub fn get_each(&self, keys: &[String]) -> Vec<Vec<f32>> {
        let m = self.map.lock().unwrap();
        keys.iter()
            .map(|k| m.get(k).cloned().unwrap_or_default())
            .collect()
    }

    pub fn add_hits(&self, n: usize) {
        self.hits.fetch_add(n, Ordering::Relaxed);
    }
    pub fn add_misses(&self, n: usize) {
        self.misses.fetch_add(n, Ordering::Relaxed);
    }
    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }

    /// Persist the cache atomically (write to a temp file, then rename).
    pub fn save(&self) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let bytes = {
            let m = self.map.lock().unwrap();
            serde_json::to_vec(&*m)?
        };
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, bytes).context("write cache")?;
        std::fs::rename(&tmp, &self.path).context("commit cache")?;
        Ok(())
    }

    /// Default cache location: `~/.cache/scry/embeddings.json`.
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Path::new(&home).join(".cache/scry/embeddings.json")
    }
}
