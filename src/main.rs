//! Command-line entry point: `cluster`, `query`, and `ask` subcommands.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use futures::stream::StreamExt;
use scry::{
    argsort_desc, cache::Cache, cluster_labels, corpus, cosine_scores, embed_code, embed_texts,
    github, l2_normalize, medoid_labels, openrouter::Client, plan_scopes, RRF_K,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "Search and cluster project READMEs",
    long_about = None
)]
struct Cli {
    /// Directory whose immediate subdirectories are projects (default: current dir).
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    /// Use a GitHub user's public repos as the corpus instead of a local dir.
    #[arg(long, global = true)]
    github: Option<String>,
    /// Use a GitHub org's repos as the corpus.
    #[arg(long, global = true)]
    github_org: Option<String>,
    /// Use a GitHub user's starred repos as the corpus.
    #[arg(long, global = true)]
    stars: Option<String>,
    /// Use an explicit comma-separated `owner/name` list as the corpus.
    #[arg(long, global = true)]
    repos: Option<String>,
    /// Include forks for --github/--github-org (default: owned, non-fork only).
    #[arg(long, global = true)]
    include_forks: bool,
    /// Re-fetch the --github corpus, bypassing the per-user disk cache.
    #[arg(long, global = true)]
    refresh: bool,
    /// OpenRouter embedding model.
    #[arg(long, global = true, default_value = "qwen/qwen3-embedding-8b")]
    model: String,
    /// Optionally truncate the embedding dimension (Matryoshka).
    #[arg(long, global = true)]
    dimensions: Option<u32>,
    /// Embedding surface: `readme` (instruction-scoped) or `code` (source files).
    #[arg(long, global = true, default_value = "readme")]
    surface: String,
    /// Model for the `code` surface.
    #[arg(long, global = true, default_value = "mistralai/codestral-embed-2505")]
    code_model: String,
    /// Pin an OpenRouter provider (e.g. Nebius). Makes embeddings deterministic
    /// and is part of the cache key so providers never mix.
    #[arg(long, global = true)]
    provider: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Embed all projects per scope and cluster them (EVoC).
    Cluster {
        /// Comma list of named scopes or `name=instruction` pairs.
        #[arg(long, default_value = "purpose,techniques,domain")]
        scopes: String,
        /// Approximate cluster-count hint.
        #[arg(long)]
        k: Option<usize>,
        /// Output directory for clusters.json.
        #[arg(long, default_value = "out")]
        out: PathBuf,
    },
    /// Rank projects by relevance to a query under one scope lens.
    Query {
        /// The query text.
        text: String,
        /// Named scope or free-text instruction lens.
        #[arg(long, default_value = "purpose")]
        scope: String,
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Emit JSON for agentic use.
        #[arg(long)]
        json: bool,
    },
    /// Dual-LLM: a planner derives scope(s)+query from a question, then fuses.
    Ask {
        /// Natural-language question.
        text: String,
        /// OpenRouter chat model used for planning.
        #[arg(long, default_value = "openai/gpt-4o-mini")]
        planner: String,
        /// How to fuse multiple scopes: rrf (rank fusion, scale-free), mean, or max.
        #[arg(long, default_value = "rrf")]
        combine: String,
        #[arg(long, default_value_t = 10)]
        top: usize,
        /// Synthesize an answer from the top hits (an LLM reads their text).
        #[arg(long)]
        answer: bool,
        /// Emit JSON for agentic use.
        #[arg(long)]
        json: bool,
    },
    /// Find near-duplicate / overlapping projects (high pairwise cosine).
    Overlap {
        /// Named scope or free-text instruction lens.
        #[arg(long, default_value = "purpose")]
        scope: String,
        /// Minimum cosine to report a pair.
        #[arg(long, default_value_t = 0.80)]
        threshold: f32,
        /// Max pairs to report.
        #[arg(long, default_value_t = 20)]
        top: usize,
        #[arg(long)]
        json: bool,
    },
    /// Evaluate retrieval against a probes file (`expected<TAB>query` per line).
    Eval {
        /// Probes TSV; lines `expected<TAB>query`, `#` comments allowed.
        file: PathBuf,
        /// Named scope or free-text instruction lens.
        #[arg(long, default_value = "purpose")]
        scope: String,
        /// Evaluate `ask` answer quality (cited + grounded) instead of retrieval.
        #[arg(long)]
        answer: bool,
        #[arg(long)]
        json: bool,
    },
    /// Run as an MCP server over stdio (exposes query/ask/overlap to agents).
    Mcp,
}

/// Parse the cluster `--scopes` spec into `(name, instruction)` pairs.
fn parse_scopes(spec: &str) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((name, instr)) = item.split_once('=') {
            out.push((name.trim().to_string(), instr.trim().to_string()));
        } else if let Some((n, instr)) = corpus::DEFAULT_SCOPES.iter().find(|(n, _)| *n == item) {
            out.push((n.to_string(), instr.to_string()));
        } else {
            bail!("unknown scope {item:?}; use a default or 'name=instruction'");
        }
    }
    Ok(out)
}

/// Lowercased tokens inside backticks (an answer's citations).
fn backtick_tokens(s: &str) -> Vec<String> {
    s.split('`')
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, t)| t.trim().to_lowercase())
        .collect()
}

/// Cluster one matrix, print the groups under `title`, and return the JSON.
fn cluster_and_emit(
    title: &str,
    mat: &[Option<Vec<f32>>],
    k: Option<usize>,
    names: &[String],
) -> Result<serde_json::Value> {
    let ClusteredRows {
        labels,
        medoids: medoid,
    } = cluster_available(mat, names, |rows| cluster_labels(rows, k))?;
    let mut groups: BTreeMap<i64, Vec<String>> = BTreeMap::new();
    for (n, lab) in names.iter().zip(&labels) {
        groups
            .entry(lab.map(|l| l as i64).unwrap_or(-1))
            .or_default()
            .push(n.clone());
    }
    println!("\n=== {title} ===");
    let mut group_json = serde_json::Map::new();
    for (key, members) in &groups {
        let mut sorted = members.clone();
        sorted.sort();
        let tag = if *key == -1 {
            "noise".to_string()
        } else {
            format!("~{}", medoid[key])
        };
        println!("  {} ({})", tag, members.len());
        println!("    {}", sorted.join(", "));
        group_json.insert(
            key.to_string(),
            serde_json::json!({ "medoid": medoid.get(key), "projects": sorted }),
        );
    }
    let label_map: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .zip(&labels)
        .map(|(n, l)| {
            (
                n.clone(),
                serde_json::json!(l.map(|x| x as i64).unwrap_or(-1)),
            )
        })
        .collect();
    Ok(serde_json::json!({ "labels": label_map, "groups": group_json }))
}

struct ClusteredRows {
    labels: Vec<Option<usize>>,
    medoids: BTreeMap<i64, String>,
}

/// Exclude absent representations before clustering, then restore original IDs.
/// The injected operation is the actual clustering boundary, also exercised by tests.
fn cluster_available(
    rows: &[Option<Vec<f32>>],
    names: &[String],
    cluster: impl FnOnce(&[Vec<f32>]) -> Result<Vec<Option<usize>>>,
) -> Result<ClusteredRows> {
    let indices: Vec<_> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().map(|_| i))
        .collect();
    let available: Vec<_> = indices.iter().map(|&i| rows[i].clone().unwrap()).collect();
    let available_names: Vec<_> = indices.iter().map(|&i| names[i].clone()).collect();
    let mut labels = vec![None; rows.len()];
    if available.is_empty() {
        return Ok(ClusteredRows {
            labels,
            medoids: BTreeMap::new(),
        });
    }
    let selected = cluster(&available)?;
    anyhow::ensure!(
        selected.len() == indices.len(),
        "cluster label count mismatch"
    );
    let medoids = medoid_labels(&available, &selected, &available_names);
    for (i, label) in indices.into_iter().zip(selected) {
        labels[i] = label;
    }
    Ok(ClusteredRows { labels, medoids })
}

/// Descending available scores followed by absent rows, preserving original IDs.
fn rank_optional(scores: &[Option<f32>]) -> Vec<usize> {
    let present: Vec<_> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|s| (i, s)))
        .collect();
    let values: Vec<_> = present.iter().map(|(_, s)| *s).collect();
    let mut order: Vec<_> = argsort_desc(&values)
        .into_iter()
        .map(|i| present[i].0)
        .collect();
    order.extend(
        scores
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.is_none().then_some(i)),
    );
    order
}

/// Fold only represented scores into the actual query/ask accumulators.
fn fold_lens(
    label: &str,
    scores: &[Option<f32>],
    combine: &str,
    detail: &mut [Vec<serde_json::Value>],
    fused: &mut [Option<f32>],
) {
    for (rank, i) in rank_optional(scores).into_iter().enumerate() {
        let Some(score) = scores[i] else { continue };
        detail[i].push(serde_json::json!({
            "scope": label, "rank": rank, "cosine": (score * 1e4).round() / 1e4,
        }));
        let value = match combine {
            "rrf" => 1.0 / (RRF_K + (rank + 1) as f32),
            _ => score,
        };
        fused[i] = Some(match (combine, fused[i]) {
            ("max", Some(previous)) => previous.max(value),
            (_, Some(previous)) => previous + value,
            (_, None) => value,
        });
    }
}

fn finish_mean(fused: &mut [Option<f32>], detail: &[Vec<serde_json::Value>]) {
    for (score, lenses) in fused.iter_mut().zip(detail) {
        if let Some(score) = score {
            *score /= lenses.len() as f32;
        }
    }
}

fn score_text(score: Option<f32>) -> String {
    score.map_or_else(|| "n/a".into(), |s| format!("{s:.4}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if !matches!(cli.surface.as_str(), "readme" | "code" | "both") {
        bail!("--surface must be readme|code|both (got {:?})", cli.surface);
    }
    let root = cli.root.clone().unwrap_or_else(corpus::default_root);
    let client = Client::from_env()?.with_provider(cli.provider.clone());
    let cache = Cache::load(Cache::default_path());

    // The code/both surfaces need a local source tree; remote sources are
    // readme/metadata only.
    let remote = cli.github.is_some()
        || cli.github_org.is_some()
        || cli.stars.is_some()
        || cli.repos.is_some();
    if remote && cli.surface != "readme" {
        bail!(
            "--surface {} needs a local source; remote (--github*/--stars/--repos) is readme-only",
            cli.surface
        );
    }
    let projects = if let Some(user) = &cli.github {
        github::fetch_user(user, cli.include_forks, cli.refresh).await?
    } else if let Some(org) = &cli.github_org {
        github::fetch_org(org, cli.include_forks, cli.refresh).await?
    } else if let Some(user) = &cli.stars {
        github::fetch_stars(user, cli.refresh).await?
    } else if let Some(list) = &cli.repos {
        let names: Vec<String> = list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        github::fetch_named(&names, cli.refresh).await?
    } else {
        corpus::discover(&root)?
    };
    if projects.is_empty() {
        bail!("no projects found");
    }

    match &cli.cmd {
        Cmd::Cluster { scopes, k, out } => {
            let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
            let mut clusters_json = serde_json::Map::new();

            if cli.surface == "code" {
                let mat =
                    embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects).await?;
                let (capped, codeless) = (mat.capped, mat.codeless());
                eprintln!(
                    "{} projects · surface=code · model={} · {capped} capped / {codeless} codeless",
                    projects.len(),
                    cli.code_model
                );
                clusters_json.insert(
                    "code".into(),
                    cluster_and_emit("surface: code", &mat.vectors, *k, &names)?,
                );
            } else {
                let scopes = parse_scopes(scopes)?;
                // For `both`, embed code once and concatenate it onto each scope.
                let code_mat = if cli.surface == "both" {
                    let m = embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects)
                        .await?;
                    let (capped, codeless) = (m.capped, m.codeless());
                    eprintln!("surface=both · code {capped} capped / {codeless} codeless");
                    Some(m)
                } else {
                    None
                };
                eprintln!(
                    "{} projects · scopes={:?} · model={}",
                    projects.len(),
                    scopes.iter().map(|(n, _)| n).collect::<Vec<_>>(),
                    cli.model
                );
                for (name, instr) in &scopes {
                    let inputs: Vec<String> = projects
                        .iter()
                        .map(|p| corpus::format_input(instr, &p.readme))
                        .collect();
                    let r_mat =
                        embed_texts(&client, &cache, &cli.model, cli.dimensions, &inputs).await?;
                    let (mat, title) = match &code_mat {
                        Some(c) => (c.with_readme(&r_mat), format!("scope+code: {name}")),
                        None => (r_mat, format!("scope: {name}")),
                    };
                    clusters_json.insert(
                        name.clone(),
                        cluster_and_emit(
                            &title,
                            &mat.into_iter().map(Some).collect::<Vec<_>>(),
                            *k,
                            &names,
                        )?,
                    );
                }
            }
            std::fs::create_dir_all(out)?;
            std::fs::write(
                out.join("clusters.json"),
                serde_json::to_vec_pretty(&clusters_json)?,
            )?;
            eprintln!(
                "\ncache: {} hit / {} miss · wrote {}/clusters.json",
                cache.hits(),
                cache.misses(),
                out.display()
            );
        }

        Cmd::Query {
            text,
            scope,
            top,
            json,
        } => {
            let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
            let (scores, scope_label) = match cli.surface.as_str() {
                "code" => {
                    let docs =
                        embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects)
                            .await?;
                    let (capped, codeless) = (docs.capped, docs.codeless());
                    eprintln!(
                        "surface=code · model={} · {capped} capped / {codeless} codeless",
                        cli.code_model
                    );
                    let qv = embed_texts(
                        &client,
                        &cache,
                        &cli.code_model,
                        cli.dimensions,
                        std::slice::from_ref(text),
                    )
                    .await?;
                    (docs.scores(&qv[0]), "code".to_string())
                }
                "both" => {
                    let instr = corpus::resolve_scope(scope);
                    let r_inputs: Vec<String> = projects
                        .iter()
                        .map(|p| corpus::format_input(&instr, &p.readme))
                        .collect();
                    let r_docs =
                        embed_texts(&client, &cache, &cli.model, cli.dimensions, &r_inputs).await?;
                    let r_q = embed_texts(
                        &client,
                        &cache,
                        &cli.model,
                        cli.dimensions,
                        &[corpus::format_input(&instr, text)],
                    )
                    .await?;
                    let c_docs =
                        embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects)
                            .await?;
                    let (capped, codeless) = (c_docs.capped, c_docs.codeless());
                    eprintln!("surface=both · code {capped} capped / {codeless} codeless");
                    let c_q = embed_texts(
                        &client,
                        &cache,
                        &cli.code_model,
                        cli.dimensions,
                        std::slice::from_ref(text),
                    )
                    .await?;
                    let mut fused = vec![None; names.len()];
                    let mut detail = vec![Vec::new(); names.len()];
                    let readme_scores = cosine_scores(&r_docs, &r_q[0])
                        .into_iter()
                        .map(Some)
                        .collect::<Vec<_>>();
                    fold_lens("readme", &readme_scores, "rrf", &mut detail, &mut fused);
                    fold_lens(
                        "code",
                        &c_docs.scores(&c_q[0]),
                        "rrf",
                        &mut detail,
                        &mut fused,
                    );
                    (fused, "both".to_string())
                }
                _ => {
                    let instr = corpus::resolve_scope(scope);
                    let doc_inputs: Vec<String> = projects
                        .iter()
                        .map(|p| corpus::format_input(&instr, &p.readme))
                        .collect();
                    let docs =
                        embed_texts(&client, &cache, &cli.model, cli.dimensions, &doc_inputs)
                            .await?;
                    let qv = embed_texts(
                        &client,
                        &cache,
                        &cli.model,
                        cli.dimensions,
                        &[corpus::format_input(&instr, text)],
                    )
                    .await?;
                    (
                        cosine_scores(&docs, &qv[0]).into_iter().map(Some).collect(),
                        instr,
                    )
                }
            };
            let order = rank_optional(&scores);

            let results: Vec<_> = order
                .iter()
                .take(*top)
                .map(|&i| serde_json::json!({ "project": names[i], "score": scores[i].map(|s| (s * 1e4).round() / 1e4) }))
                .collect();
            if *json {
                println!(
                    "{}",
                    serde_json::json!({ "query": text, "scope": scope_label, "results": results })
                );
            } else {
                eprintln!("query: {text:?}\nscope: {scope_label:?}");
                for &i in order.iter().take(*top) {
                    println!("  {}  {}", score_text(scores[i]), names[i]);
                }
            }
            eprintln!("cache: {} hit / {} miss", cache.hits(), cache.misses());
        }

        Cmd::Ask {
            text,
            planner,
            combine,
            top,
            answer,
            json,
        } => {
            if !matches!(combine.as_str(), "rrf" | "mean" | "max") {
                bail!("--combine must be rrf|mean|max (got {:?})", combine);
            }
            let (scopes, query) = plan_scopes(&client, planner, text).await?;
            let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
            let n = names.len();
            eprintln!(
                "planner={planner} · {} scope(s) · combine={combine} · query={query:?}",
                scopes.len()
            );

            let use_readme = cli.surface != "code";
            let use_code = cli.surface != "readme";
            let mut fused = vec![None; n];
            let mut per_scope_detail: Vec<Vec<serde_json::Value>> = vec![Vec::new(); n];

            if use_readme {
                for instr in &scopes {
                    let doc_inputs: Vec<String> = projects
                        .iter()
                        .map(|p| corpus::format_input(instr, &p.readme))
                        .collect();
                    let docs =
                        embed_texts(&client, &cache, &cli.model, cli.dimensions, &doc_inputs)
                            .await?;
                    let qv = embed_texts(
                        &client,
                        &cache,
                        &cli.model,
                        cli.dimensions,
                        &[corpus::format_input(instr, &query)],
                    )
                    .await?;
                    let s = cosine_scores(&docs, &qv[0])
                        .into_iter()
                        .map(Some)
                        .collect::<Vec<_>>();
                    fold_lens(instr, &s, combine, &mut per_scope_detail, &mut fused);
                }
            }
            if use_code {
                let c_docs =
                    embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects).await?;
                let (capped, codeless) = (c_docs.capped, c_docs.codeless());
                eprintln!("code lens · {capped} capped / {codeless} codeless");
                let c_q = embed_texts(
                    &client,
                    &cache,
                    &cli.code_model,
                    cli.dimensions,
                    std::slice::from_ref(&query),
                )
                .await?;
                let s = c_docs.scores(&c_q[0]);
                fold_lens("<code>", &s, combine, &mut per_scope_detail, &mut fused);
            }

            if combine == "mean" {
                finish_mean(&mut fused, &per_scope_detail);
            }

            let order = rank_optional(&fused);
            let results: Vec<_> = order
                .iter()
                .take(*top)
                .map(|&i| {
                    serde_json::json!({
                        "project": names[i],
                        "score": fused[i].map(|s| (s * 1e5).round() / 1e5),
                        "scopes": per_scope_detail[i],
                    })
                })
                .collect();
            // Optional synthesis: an LLM reads the top hits' text and answers,
            // citing project names. Turns retrieval into question-answering.
            let synthesized = if *answer {
                let ctx: String = order
                    .iter()
                    .take(*top)
                    .map(|&i| {
                        let snip: String = projects[i].readme.chars().take(1500).collect();
                        format!("## {}\n{snip}", names[i])
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let sys = "Answer the user's question about a collection of software \
                    projects using ONLY the provided summaries. Cite project names in \
                    backticks. Be concise (a few sentences). If none fit, say so.";
                Some(
                    client
                        .chat(
                            planner,
                            sys,
                            &format!("Question: {text}\n\nProjects:\n{ctx}"),
                        )
                        .await?,
                )
            } else {
                None
            };

            if *json {
                println!(
                    "{}",
                    serde_json::json!({
                        "question": text, "plan": { "scopes": scopes, "query": query },
                        "combine": combine, "answer": synthesized, "results": results,
                    })
                );
            } else {
                eprintln!("Q: {text}");
                if use_readme {
                    for s in &scopes {
                        eprintln!("  lens: {s}");
                    }
                }
                if use_code {
                    eprintln!("  lens: <code>");
                }
                for &i in order.iter().take(*top) {
                    println!("  {}  {}", score_text(fused[i]), names[i]);
                }
                if let Some(a) = &synthesized {
                    println!("\n{a}");
                }
            }
            eprintln!("cache: {} hit / {} miss", cache.hits(), cache.misses());
        }

        Cmd::Overlap {
            scope,
            threshold,
            top,
            json,
        } => {
            let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
            let instr = corpus::resolve_scope(scope);
            let inputs: Vec<String> = projects
                .iter()
                .map(|p| corpus::format_input(&instr, &p.readme))
                .collect();
            let mat = embed_texts(&client, &cache, &cli.model, cli.dimensions, &inputs).await?;
            let norm: Vec<Vec<f32>> = mat.iter().map(|v| l2_normalize(v)).collect();
            let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
            for i in 0..norm.len() {
                for j in (i + 1)..norm.len() {
                    let s: f32 = norm[i].iter().zip(&norm[j]).map(|(a, b)| a * b).sum();
                    if s >= *threshold {
                        pairs.push((s, i, j));
                    }
                }
            }
            pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
            pairs.truncate(*top);
            if *json {
                let arr: Vec<_> = pairs
                    .iter()
                    .map(|(s, i, j)| {
                        serde_json::json!({"a": names[*i], "b": names[*j], "score": (s * 1e4).round() / 1e4})
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({"scope": instr, "threshold": threshold, "pairs": arr})
                );
            } else {
                eprintln!("overlap (scope {instr:?}, threshold {threshold})");
                if pairs.is_empty() {
                    eprintln!("  no pairs above threshold");
                }
                for (s, i, j) in &pairs {
                    println!("  {:.4}  {} ~ {}", s, names[*i], names[*j]);
                }
            }
            eprintln!("cache: {} hit / {} miss", cache.hits(), cache.misses());
        }

        Cmd::Eval {
            file,
            scope,
            answer,
            json,
        } => {
            let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
            let content = std::fs::read_to_string(file)?;
            let probes: Vec<(String, String)> = content
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .filter_map(|l| {
                    l.split_once('\t')
                        .map(|(e, q)| (e.trim().to_string(), q.trim().to_string()))
                })
                .collect();
            if probes.is_empty() {
                bail!("no probes in {}", file.display());
            }

            if *answer {
                // Answer-quality eval: run the ask flow per probe and check, from
                // the synthesized answer's backtick citations, whether it named the
                // expected project (cited) and whether every cited project is one it
                // actually retrieved (grounded = no invented projects).
                // Probes run concurrently (bounded); the shared &Cache is
                // interior-mutable so the ask flows don't serialize on it.
                let detail: Vec<(String, bool, bool)> = futures::stream::iter(probes.iter())
                    .map(|(expected, question)| {
                        let (client, cache, projects, model, dims) =
                            (&client, &cache, &projects, &cli.model, cli.dimensions);
                        async move {
                            let res = scry::ask(
                                client,
                                cache,
                                model,
                                dims,
                                "openai/gpt-4o-mini",
                                projects,
                                question,
                                8,
                                true,
                            )
                            .await?;
                            let toks = backtick_tokens(&res.answer.unwrap_or_default());
                            let cited = toks.contains(&expected.to_lowercase());
                            let grounded = res
                                .ranked
                                .iter()
                                .any(|(nm, _)| toks.contains(&nm.to_lowercase()));
                            anyhow::Ok((expected.clone(), cited, grounded))
                        }
                    })
                    .buffered(6)
                    .collect::<Vec<_>>()
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                let n = probes.len();
                let nf = n as f64;
                let cited = detail.iter().filter(|(_, c, _)| *c).count();
                let grounded = detail.iter().filter(|(_, _, g)| *g).count();
                if *json {
                    let per: Vec<_> = detail
                        .iter()
                        .map(|(e, c, g)| serde_json::json!({ "expected": e, "cited": c, "grounded": g }))
                        .collect();
                    println!(
                        "{}",
                        serde_json::json!({ "n": n, "cited": cited, "grounded": grounded, "probes": per })
                    );
                } else {
                    eprintln!("ask eval (n={n})");
                    println!(
                        "cited {cited}/{n} ({:.0}%)  grounded {grounded}/{n} ({:.0}%)",
                        100.0 * cited as f64 / nf,
                        100.0 * grounded as f64 / nf
                    );
                    for (e, c, g) in &detail {
                        if !c {
                            println!("  not cited: {e} (grounded={g})");
                        }
                    }
                }
                eprintln!("cache: {} hit / {} miss", cache.hits(), cache.misses());
                return Ok(());
            }

            let instr = corpus::resolve_scope(scope);
            let inputs: Vec<String> = projects
                .iter()
                .map(|p| corpus::format_input(&instr, &p.readme))
                .collect();
            let docs = embed_texts(&client, &cache, &cli.model, cli.dimensions, &inputs).await?;
            let qinputs: Vec<String> = probes
                .iter()
                .map(|(_, q)| corpus::format_input(&instr, q))
                .collect();
            let qv = embed_texts(&client, &cache, &cli.model, cli.dimensions, &qinputs).await?;
            let ranks: Vec<(String, Option<usize>)> = probes
                .iter()
                .enumerate()
                .map(|(idx, (expected, _))| {
                    let scores = cosine_scores(&docs, &qv[idx]);
                    let rank = argsort_desc(&scores)
                        .iter()
                        .position(|&i| names[i] == *expected)
                        .map(|p| p + 1);
                    (expected.clone(), rank)
                })
                .collect();
            let n = probes.len();
            let nf = n as f64;
            let top1 = ranks.iter().filter(|(_, r)| *r == Some(1)).count();
            let top3 = ranks
                .iter()
                .filter(|(_, r)| r.is_some_and(|r| r <= 3))
                .count();
            let mrr: f64 = ranks
                .iter()
                .filter_map(|(_, r)| r.map(|r| 1.0 / r as f64))
                .sum::<f64>()
                / nf;
            if *json {
                let per: Vec<_> = ranks
                    .iter()
                    .map(|(e, r)| serde_json::json!({ "expected": e, "rank": r }))
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({ "scope": instr, "n": n, "top1": top1, "top3": top3, "mrr": mrr, "probes": per })
                );
            } else {
                eprintln!("eval (scope {instr:?}, n={n})");
                println!(
                    "top-1 {top1}/{n} ({:.0}%)  top-3 {top3}/{n} ({:.0}%)  MRR {mrr:.3}",
                    100.0 * top1 as f64 / nf,
                    100.0 * top3 as f64 / nf
                );
                for (e, r) in &ranks {
                    if r.is_none_or(|r| r > 3) {
                        println!("  miss: {e} (rank {r:?})");
                    }
                }
            }
            eprintln!("cache: {} hit / {} miss", cache.hits(), cache.misses());
        }

        Cmd::Mcp => {
            scry::mcp::serve(&projects, &client, &cache, &cli.model, cli.dimensions).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry::code;

    #[tokio::test]
    async fn cached_code_absence_stays_distinct_through_cli_ranking_and_clustering() {
        // This test binary never needs real credentials. Every source chunk is
        // preseeded below before calling the production gather/embedding path.
        std::env::set_var("OPENROUTER_API_KEY", "offline-synthetic-unused");
        let root = std::env::temp_dir().join(format!(
            "scry-missing-code-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let model = "offline-fixture";
        let cache = Cache::load(root.join("cache.json"));
        let mut projects = Vec::new();
        for (name, source, vector) in [
            ("negative", Some("fn negative() {}"), vec![-1.0, 0.0]),
            ("absent", None, vec![]),
            ("zero", Some("fn zero() {}"), vec![0.0, 0.0]),
        ] {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            if let Some(text) = source {
                std::fs::write(dir.join("lib.rs"), text).unwrap();
                cache.insert(Cache::key(model, None, None, text), vector);
            }
            projects.push(corpus::Project {
                name: name.into(),
                dir,
                readme: "fixture".into(),
            });
        }
        let keys: Vec<_> = projects
            .iter()
            .flat_map(|p| code::list_source_files(&p.dir))
            .flat_map(|f| code::chunk_text(&std::fs::read_to_string(f).unwrap(), code::CHUNK_CHARS))
            .map(|text| Cache::key(model, None, None, &text))
            .collect();
        assert_eq!(keys.len(), 2);
        assert_eq!(cache.contains_each(&keys), [true, true]);
        let client = Client::from_env().unwrap();
        let code = embed_code(&client, &cache, model, None, &projects)
            .await
            .unwrap();
        assert_eq!((cache.hits(), cache.misses()), (2, 0));
        assert_eq!((code.capped, code.codeless()), (0, 1));
        assert_eq!(
            code.vectors,
            [Some(vec![-1.0, 0.0]), None, Some(vec![0.0, 0.0])]
        );
        let scores = code.scores(&[1.0, 0.0]);
        assert_eq!(scores, [Some(-1.0), None, Some(0.0)]);
        assert_eq!(rank_optional(&scores), [2, 0, 1]);
        assert_eq!(
            serde_json::json!(scores),
            serde_json::json!([-1.0, null, 0.0])
        );
        assert_eq!(score_text(scores[1]), "n/a");
        let names = projects.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        let ClusteredRows { labels, medoids } = cluster_available(&code.vectors, &names, |rows| {
            assert_eq!(rows, [vec![-1.0, 0.0], vec![0.0, 0.0]]);
            Ok(vec![Some(4), Some(9)])
        })
        .unwrap();
        assert_eq!(labels, [Some(4), None, Some(9)]);
        assert_eq!(medoids[&4], "negative");
        assert_eq!(medoids[&9], "zero");
        assert_eq!(
            code.with_readme(&vec![vec![3.0, 4.0]; 3]),
            [
                vec![0.6, 0.8, -1.0, 0.0],
                vec![0.6, 0.8, 0.0, 0.0],
                vec![0.6, 0.8, 0.0, 0.0]
            ]
        );
        let all_missing = embed_code(
            &client,
            &cache,
            model,
            None,
            std::slice::from_ref(&projects[1]),
        )
        .await
        .unwrap();
        assert_eq!(all_missing.vectors, [None]);
        assert_eq!(cache.misses(), 0);
        let ClusteredRows { labels, medoids } =
            cluster_available(&all_missing.vectors, &["absent".into()], |_| {
                panic!("no rows to cluster")
            })
            .unwrap();
        assert_eq!(labels, [None]);
        assert!(medoids.is_empty());
        let emitted = cluster_and_emit(
            "missing fixture",
            &all_missing.vectors,
            None,
            &["absent".into()],
        )
        .unwrap();
        assert_eq!(emitted["labels"]["absent"], -1);
        assert_eq!(
            emitted["groups"]["-1"]["projects"],
            serde_json::json!(["absent"])
        );
        assert!(emitted["groups"]["-1"]["medoid"].is_null());
        assert_eq!(all_missing.with_readme(&[vec![3.0, 4.0]]), [vec![0.6, 0.8]]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn query_and_ask_fusion_skip_absent_votes_and_keep_project_identity() {
        let readme = [Some(-0.4), Some(0.2), Some(0.5)];
        let code = [Some(-1.0), None, Some(0.0)];
        for combine in ["rrf", "mean", "max"] {
            let mut detail = vec![vec![]; 3];
            let mut fused = vec![None; 3];
            fold_lens("readme", &readme, combine, &mut detail, &mut fused);
            fold_lens("code", &code, combine, &mut detail, &mut fused);
            if combine == "mean" {
                finish_mean(&mut fused, &detail);
            }
            assert_eq!(
                detail[1].len(),
                1,
                "missing code must not vote or alter the mean divisor"
            );
            assert_eq!(detail[1][0]["scope"], "readme");
            assert_eq!(detail[0][1]["rank"], 1);
            assert_eq!(detail[2][1]["rank"], 0);
            let expected = match combine {
                "rrf" => [1.0 / 63.0 + 1.0 / 62.0, 1.0 / 62.0, 2.0 / 61.0],
                "mean" => [-0.7, 0.2, 0.25],
                _ => [-0.4, 0.2, 0.5],
            };
            for (actual, expected) in fused.iter().zip(expected) {
                assert!((actual.unwrap() - expected).abs() < 1e-6);
            }
            let mut only_code = vec![None; 3];
            let mut detail = vec![vec![]; 3];
            fold_lens("code", &code, combine, &mut detail, &mut only_code);
            if combine == "mean" {
                finish_mean(&mut only_code, &detail);
            }
            assert_eq!(only_code[1], None);
            assert_eq!(rank_optional(&only_code), [2, 0, 1]);
            assert!(detail[1].is_empty());
        }
    }
}
