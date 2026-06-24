//! # scry
//!
//! Scry your project corpus: scoped, instruction-conditioned embeddings of
//! dev-project docs, with EVoC clustering and a dual-LLM "ask" front-end, all
//! over OpenRouter. Reuses [`clump`] for clustering and talks to OpenRouter
//! directly via [`openrouter`].
//!
//! The central idea is the *scope*: an instruction lens (purpose / techniques /
//! domain, or any free-text instruction) prepended to each README before
//! embedding, so the same corpus ranks and groups differently per lens.
//!
//! Pipeline:
//! 1. [`corpus::discover`] finds projects with READMEs.
//! 2. [`embed_texts`] embeds them per scope, concurrently, through a disk
//!    [`cache::Cache`] so unchanged inputs are never re-embedded.
//! 3. [`cluster_labels`] groups one scope's vectors with EVoC; [`medoid_labels`]
//!    names each cluster by its most central project.
//! 4. [`rank`] scores projects against a query under one lens; [`rrf_fuse`]
//!    combines several lenses by Reciprocal Rank Fusion (scale-free, the right
//!    way to merge rankings whose score ranges differ per lens).

pub mod cache;
pub mod code;
pub mod corpus;
pub mod github;
pub mod mcp;
pub mod openrouter;

use anyhow::{Context, Result};
use cache::Cache;
use clump::{EVoC, EVoCParams};
use corpus::Project;
use futures::stream::{self, StreamExt};
use openrouter::Client;

const BATCH_SIZE: usize = 32;
const CONCURRENCY: usize = 6;
/// Reciprocal-rank-fusion constant; the standard default.
pub const RRF_K: f32 = 60.0;

/// Embed `texts`, returning one vector per input, embedding only cache misses.
///
/// Misses are batched and the batches run concurrently (bounded by
/// [`CONCURRENCY`]). The cache is consulted per unique text, so repeated inputs
/// cost nothing, and it is saved once after the misses resolve.
pub async fn embed_texts(
    client: &Client,
    cache: &Cache,
    model: &str,
    dimensions: Option<u32>,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let provider = client.provider();
    let keys: Vec<String> = texts
        .iter()
        .map(|t| Cache::key(model, provider, dimensions, t))
        .collect();
    let present = cache.contains_each(&keys);
    let hits = present.iter().filter(|&&p| p).count();
    cache.add_hits(hits);
    cache.add_misses(keys.len() - hits);

    let mut miss_texts: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for ((t, k), p) in texts.iter().zip(&keys).zip(&present) {
        if !*p && seen.insert(k.clone()) {
            miss_texts.push(t.clone());
        }
    }

    if !miss_texts.is_empty() {
        let batches: Vec<Vec<String>> = miss_texts.chunks(BATCH_SIZE).map(|c| c.to_vec()).collect();
        let total = batches.len();
        // Save the cache as batches land (every 8), so a long or interrupted run
        // persists progress and a re-run resumes from the cache instead of
        // re-embedding everything.
        let mut stream = stream::iter(batches.iter())
            .map(|b| async move { client.embed(model, dimensions, b).await.map(|v| (b, v)) })
            .buffered(CONCURRENCY);
        let mut done = 0usize;
        while let Some(res) = stream.next().await {
            let (batch, vecs) = res?;
            for (t, v) in batch.iter().zip(vecs) {
                cache.insert(Cache::key(model, provider, dimensions, t), v);
            }
            done += 1;
            if done.is_multiple_of(8) {
                cache.save().context("save cache")?;
                eprintln!("  embedded {done}/{total} batches");
            }
        }
        cache.save().context("save cache")?;
    }

    Ok(cache.get_each(&keys))
}

/// Read a file with a hard timeout, returning `None` on timeout or error so the
/// caller skips it. Guards against iCloud-dataless files (and fifos, stalled
/// mounts) that would otherwise block `read_to_string` indefinitely.
async fn read_file_bounded(path: std::path::PathBuf) -> Option<String> {
    let fut = tokio::task::spawn_blocking(move || std::fs::read_to_string(path));
    match tokio::time::timeout(std::time::Duration::from_secs(5), fut).await {
        Ok(Ok(Ok(s))) => Some(s),
        _ => None,
    }
}

/// Embed the `code` surface: one mean-pooled vector per project, built from its
/// source-file chunks (embedded with `model`, typically a code model). Chunks
/// are embedded through the same cache. Returns `(vectors, capped, codeless)`:
/// how many projects hit the per-project chunk cap, and how many had no source
/// at all (those get a zero vector and sort last / cluster as noise).
pub async fn embed_code(
    client: &Client,
    cache: &Cache,
    model: &str,
    dimensions: Option<u32>,
    projects: &[Project],
) -> Result<(Vec<Vec<f32>>, usize, usize)> {
    let mut all_chunks: Vec<String> = Vec::new();
    let mut counts: Vec<usize> = Vec::with_capacity(projects.len());
    let mut capped = 0usize;
    eprintln!("gathering source from {} projects...", projects.len());
    for p in projects {
        let mut chunks: Vec<String> = Vec::new();
        let mut was_capped = false;
        for f in code::list_source_files(&p.dir) {
            if chunks.len() >= code::MAX_CHUNKS {
                was_capped = true;
                break;
            }
            let Some(text) = read_file_bounded(f).await else {
                continue; // timeout (e.g. iCloud-dataless) or read error -> skip
            };
            for c in code::chunk_text(&text, code::CHUNK_CHARS) {
                if chunks.len() >= code::MAX_CHUNKS {
                    was_capped = true;
                    break;
                }
                chunks.push(c);
            }
        }
        if was_capped {
            capped += 1;
        }
        counts.push(chunks.len());
        all_chunks.extend(chunks);
    }
    eprintln!("gathered {} chunks; embedding...", all_chunks.len());
    if all_chunks.is_empty() {
        anyhow::bail!("no source files found under any project");
    }
    let chunk_vecs = embed_texts(client, cache, model, dimensions, &all_chunks).await?;
    let dim = chunk_vecs[0].len();

    let mut vectors = Vec::with_capacity(projects.len());
    let mut codeless = 0usize;
    let mut offset = 0usize;
    for &count in &counts {
        if count == 0 {
            codeless += 1;
            vectors.push(vec![0.0f32; dim]);
            continue;
        }
        let mut pooled = vec![0.0f32; dim];
        for v in &chunk_vecs[offset..offset + count] {
            for (acc, x) in pooled.iter_mut().zip(v) {
                *acc += x;
            }
        }
        for acc in &mut pooled {
            *acc /= count as f32;
        }
        vectors.push(pooled);
        offset += count;
    }
    Ok((vectors, capped, codeless))
}

/// L2-normalize a vector (zero vectors are left unchanged).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Cosine similarity of every document against the query (all are normalized
/// internally). Returns one score per document, in document order.
pub fn cosine_scores(docs: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
    let q = l2_normalize(query);
    docs.iter()
        .map(|d| {
            let dn = l2_normalize(d);
            dn.iter().zip(&q).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// Cluster one scope's embedding vectors with EVoC. Returns a label per vector;
/// `None` is noise. When `k` is given, the hierarchy is cut to approximately
/// that many clusters.
pub fn cluster_labels(vecs: &[Vec<f32>], k: Option<usize>) -> Result<Vec<Option<usize>>> {
    let normalized: Vec<Vec<f32>> = vecs.iter().map(|v| l2_normalize(v)).collect();
    // Fixed seed so clustering is reproducible run-to-run (EVoC defaults to a
    // random seed, which makes an agent see different clusters each call).
    let params = EVoCParams {
        min_cluster_size: 3,
        seed: Some(0),
        ..Default::default()
    };
    let mut model = EVoC::new(params);
    match k {
        Some(k) => {
            model.fit_predict(&normalized).context("evoc fit")?;
            Ok(model
                .layer_for_n_clusters(k)
                .context("evoc layer for k")?
                .assignments)
        }
        None => model.fit_predict(&normalized).context("evoc fit_predict"),
    }
}

/// Label each cluster by its medoid: the project nearest the cluster centroid.
/// `labels` aligns with `names`; `None` labels (noise) map to `usize::MAX`.
pub fn medoid_labels(
    vecs: &[Vec<f32>],
    labels: &[Option<usize>],
    names: &[String],
) -> std::collections::BTreeMap<i64, String> {
    use std::collections::BTreeMap;
    let normalized: Vec<Vec<f32>> = vecs.iter().map(|v| l2_normalize(v)).collect();
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, lab) in labels.iter().enumerate() {
        let key = lab.map(|l| l as i64).unwrap_or(-1);
        groups.entry(key).or_default().push(i);
    }
    let mut out = BTreeMap::new();
    for (key, rows) in groups {
        let dim = normalized[rows[0]].len();
        let mut centroid = vec![0.0f32; dim];
        for &i in &rows {
            for (c, x) in centroid.iter_mut().zip(&normalized[i]) {
                *c += x;
            }
        }
        for c in &mut centroid {
            *c /= rows.len() as f32;
        }
        let best = rows
            .iter()
            .max_by(|&&a, &&b| {
                let sa: f32 = normalized[a]
                    .iter()
                    .zip(&centroid)
                    .map(|(x, y)| x * y)
                    .sum();
                let sb: f32 = normalized[b]
                    .iter()
                    .zip(&centroid)
                    .map(|(x, y)| x * y)
                    .sum();
                sa.total_cmp(&sb)
            })
            .copied()
            .unwrap_or(rows[0]);
        out.insert(key, names[best].clone());
    }
    out
}

/// Fuse per-scope rankings by Reciprocal Rank Fusion. `per_scope_ranks[s][i]` is
/// project `i`'s 0-based rank under scope `s`. Returns a fused score per project.
pub fn rrf_fuse(per_scope_ranks: &[Vec<usize>], n: usize) -> Vec<f32> {
    let mut fused = vec![0.0f32; n];
    for ranks in per_scope_ranks {
        for (i, &r) in ranks.iter().enumerate() {
            fused[i] += 1.0 / (RRF_K + r as f32);
        }
    }
    fused
}

/// Rank indices by descending score.
pub fn argsort_desc(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx
}

/// 0-based rank of each item by descending score (`rank[i]` = position of `i`).
pub fn ranks_desc(scores: &[f32]) -> Vec<usize> {
    let mut r = vec![0usize; scores.len()];
    for (pos, &i) in argsort_desc(scores).iter().enumerate() {
        r[i] = pos;
    }
    r
}

/// Concatenate two aligned vector sets, L2-normalizing each part first so neither
/// surface dominates by magnitude. Used for the `both` cluster surface (joint
/// readme+code feature space).
pub fn concat_normalized(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let mut v = l2_normalize(x);
            v.extend(l2_normalize(y));
            v
        })
        .collect()
}

/// Dual-LLM planner: turn a natural-language question into 1-3 scope lenses plus
/// a query string. The model emits structured intent only; the caller executes
/// the read-only retrieval. Returns `(scopes, query)`.
pub async fn plan_scopes(
    client: &Client,
    planner: &str,
    question: &str,
) -> Result<(Vec<String>, String)> {
    let system = "You turn a question about a collection of software projects into a \
        retrieval plan. Output ONLY JSON: {\"scopes\": [\"<lens instruction>\", ...], \
        \"query\": \"<search text>\"}. Each scope is a short instruction naming the \
        ASPECT to match on (its purpose, the technique it uses, its domain, the data \
        structure, etc.) phrased like 'Represent this project by ...'. Use 1-3 \
        complementary scopes when the question spans aspects, else 1. The query is the \
        thing to look for under those lenses.";
    let content = client.chat_json(planner, system, question).await?;
    let parsed: serde_json::Value = serde_json::from_str(&content).context("parse plan JSON")?;
    let scopes: Vec<String> = parsed["scopes"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();
    let scopes = if scopes.is_empty() {
        vec!["Represent this project by its core purpose.".to_string()]
    } else {
        scopes
    };
    let query = parsed["query"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(question)
        .to_string();
    Ok((scopes, query))
}

/// Result of the [`ask`] flow.
pub struct AskResult {
    pub scopes: Vec<String>,
    pub query: String,
    /// Top projects with fused scores, best first.
    pub ranked: Vec<(String, f32)>,
    /// Synthesized answer (only when `synthesize`).
    pub answer: Option<String>,
}

/// The full ask flow: a planner derives scope lenses, each retrieves, rankings
/// are RRF-fused, and (optionally) an LLM synthesizes an answer over the top
/// hits. Shared by the `ask` command, the MCP `scry_ask` tool, and the eval.
#[allow(clippy::too_many_arguments)]
pub async fn ask(
    client: &Client,
    cache: &Cache,
    model: &str,
    dimensions: Option<u32>,
    planner: &str,
    projects: &[Project],
    question: &str,
    top: usize,
    synthesize: bool,
) -> Result<AskResult> {
    let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
    let n = names.len();
    let (scopes, query) = plan_scopes(client, planner, question).await?;

    let mut rank_lists: Vec<Vec<usize>> = Vec::new();
    for instr in &scopes {
        let inputs: Vec<String> = projects
            .iter()
            .map(|p| corpus::format_input(instr, &p.readme))
            .collect();
        let docs = embed_texts(client, cache, model, dimensions, &inputs).await?;
        let qv = embed_texts(
            client,
            cache,
            model,
            dimensions,
            &[corpus::format_input(instr, &query)],
        )
        .await?;
        rank_lists.push(ranks_desc(&cosine_scores(&docs, &qv[0])));
    }
    let fused = rrf_fuse(&rank_lists, n);
    let order = argsort_desc(&fused);
    let ranked: Vec<(String, f32)> = order
        .iter()
        .take(top)
        .map(|&i| (names[i].clone(), fused[i]))
        .collect();

    let answer = if synthesize {
        let ctx: String = order
            .iter()
            .take(top)
            .map(|&i| {
                let snip: String = projects[i].readme.chars().take(1500).collect();
                format!("## {}\n{snip}", names[i])
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let sys = "Answer the user's question about a collection of software projects using \
            ONLY the provided summaries. Cite project names in backticks. Be concise. If none \
            fit, say so.";
        Some(
            client
                .chat(
                    planner,
                    sys,
                    &format!("Question: {question}\n\nProjects:\n{ctx}"),
                )
                .await?,
        )
    } else {
        None
    };
    Ok(AskResult {
        scopes,
        query,
        ranked,
        answer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_and_zero() {
        let v = l2_normalize(&[3.0, 4.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert_eq!(l2_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_scores_identical_orthogonal_and_45deg() {
        let docs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let s = cosine_scores(&docs, &[1.0, 0.0]);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(s[1].abs() < 1e-6);
        assert!((s[2] - 1.0 / 2f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn argsort_and_ranks_agree() {
        let scores = [0.1, 0.5, 0.9];
        assert_eq!(argsort_desc(&scores), vec![2, 1, 0]);
        assert_eq!(ranks_desc(&scores), vec![2, 1, 0]);
    }

    #[test]
    fn rrf_rewards_consistent_top() {
        // index 0 is rank 0 under both scopes; index 2 is worst under both.
        let fused = rrf_fuse(&[vec![0, 1, 2], vec![0, 1, 2]], 3);
        assert!(fused[0] > fused[1] && fused[1] > fused[2]);
        // a split-decision item beats a consistently-mediocre one is NOT asserted;
        // here we only check monotonicity under agreement.
        assert!((fused[0] - 2.0 / (RRF_K + 0.0)).abs() < 1e-6);
    }

    #[test]
    fn concat_normalizes_each_half() {
        let a = vec![vec![3.0, 4.0]];
        let b = vec![vec![0.0, 5.0]];
        let c = concat_normalized(&a, &b);
        assert_eq!(c[0].len(), 4);
        assert!((c[0][0] - 0.6).abs() < 1e-6 && (c[0][1] - 0.8).abs() < 1e-6);
        assert!(c[0][2].abs() < 1e-6 && (c[0][3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn medoid_is_the_central_member() {
        // index 2 sits between 0 and 1, so it is nearest the centroid.
        let vecs = vec![vec![1.0, 0.0], vec![0.6, 0.8], vec![0.92, 0.39]];
        let labels = vec![Some(0), Some(0), Some(0)];
        let names = ["a", "b", "c"].map(String::from).to_vec();
        let m = medoid_labels(&vecs, &labels, &names);
        assert_eq!(m[&0], "c");
    }
}
