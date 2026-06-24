//! Command-line entry point: `cluster`, `query`, and `ask` subcommands.

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use futures::stream::StreamExt;
use scry::{
    argsort_desc, cache::Cache, cluster_labels, concat_normalized, corpus, cosine_scores,
    embed_code, embed_texts, github, l2_normalize, medoid_labels, openrouter::Client, plan_scopes,
    ranks_desc, rrf_fuse,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    about = "Scoped README embeddings + EVoC clustering + dual-LLM ask, over OpenRouter",
    long_about = None
)]
struct Cli {
    /// Directory whose immediate subdirectories are projects (default: ~/Documents/dev).
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
    mat: &[Vec<f32>],
    k: Option<usize>,
    names: &[String],
) -> Result<serde_json::Value> {
    let labels = cluster_labels(mat, k)?;
    let medoid = medoid_labels(mat, &labels, names);
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

/// Fold one lens's scores into the `ask` accumulators: append the per-project
/// detail, push the rank list, and (for max/mean) update the running fused score.
#[allow(clippy::too_many_arguments)]
fn fold_lens(
    label: &str,
    scores: &[f32],
    n: usize,
    combine: &str,
    detail: &mut [Vec<serde_json::Value>],
    rank_lists: &mut Vec<Vec<usize>>,
    fused: &mut [f32],
) {
    let mut rank_of = vec![0usize; n];
    for (r, &i) in argsort_desc(scores).iter().enumerate() {
        rank_of[i] = r;
    }
    for (i, d) in detail.iter_mut().enumerate() {
        d.push(serde_json::json!({
            "scope": label,
            "rank": rank_of[i],
            "cosine": (scores[i] * 1e4).round() / 1e4,
        }));
        match combine {
            "max" => fused[i] = fused[i].max(scores[i]),
            "mean" => fused[i] += scores[i],
            _ => {}
        }
    }
    rank_lists.push(rank_of);
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
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
                let (mat, capped, codeless) =
                    embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects).await?;
                eprintln!(
                    "{} projects · surface=code · model={} · {capped} capped / {codeless} codeless",
                    projects.len(),
                    cli.code_model
                );
                clusters_json.insert(
                    "code".into(),
                    cluster_and_emit("surface: code", &mat, *k, &names)?,
                );
            } else {
                let scopes = parse_scopes(scopes)?;
                // For `both`, embed code once and concatenate it onto each scope.
                let code_mat = if cli.surface == "both" {
                    let (m, capped, codeless) =
                        embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects)
                            .await?;
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
                        Some(c) => (concat_normalized(&r_mat, c), format!("scope+code: {name}")),
                        None => (r_mat, format!("scope: {name}")),
                    };
                    clusters_json.insert(name.clone(), cluster_and_emit(&title, &mat, *k, &names)?);
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
                    let (docs, capped, codeless) =
                        embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects)
                            .await?;
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
                    (cosine_scores(&docs, &qv[0]), "code".to_string())
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
                    let (c_docs, capped, codeless) =
                        embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects)
                            .await?;
                    eprintln!("surface=both · code {capped} capped / {codeless} codeless");
                    let c_q = embed_texts(
                        &client,
                        &cache,
                        &cli.code_model,
                        cli.dimensions,
                        std::slice::from_ref(text),
                    )
                    .await?;
                    let r_ranks = ranks_desc(&cosine_scores(&r_docs, &r_q[0]));
                    let c_ranks = ranks_desc(&cosine_scores(&c_docs, &c_q[0]));
                    (
                        rrf_fuse(&[r_ranks, c_ranks], names.len()),
                        "both".to_string(),
                    )
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
                    (cosine_scores(&docs, &qv[0]), instr)
                }
            };
            let order = argsort_desc(&scores);

            let results: Vec<_> = order
                .iter()
                .take(*top)
                .map(|&i| serde_json::json!({ "project": names[i], "score": (scores[i] * 1e4).round() / 1e4 }))
                .collect();
            if *json {
                println!(
                    "{}",
                    serde_json::json!({ "query": text, "scope": scope_label, "results": results })
                );
            } else {
                eprintln!("query: {text:?}\nscope: {scope_label:?}");
                for &i in order.iter().take(*top) {
                    println!("  {:.4}  {}", scores[i], names[i]);
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
            let (scopes, query) = plan_scopes(&client, planner, text).await?;
            let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
            let n = names.len();
            eprintln!(
                "planner={planner} · {} scope(s) · combine={combine} · query={query:?}",
                scopes.len()
            );

            let use_readme = cli.surface != "code";
            let use_code = cli.surface != "readme";
            let mut fused = vec![0.0f32; n];
            let mut per_scope_detail: Vec<Vec<serde_json::Value>> = vec![Vec::new(); n];
            let mut rank_lists: Vec<Vec<usize>> = Vec::new();

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
                    let s = cosine_scores(&docs, &qv[0]);
                    fold_lens(
                        instr,
                        &s,
                        n,
                        combine,
                        &mut per_scope_detail,
                        &mut rank_lists,
                        &mut fused,
                    );
                }
            }
            if use_code {
                let (c_docs, capped, codeless) =
                    embed_code(&client, &cache, &cli.code_model, cli.dimensions, &projects).await?;
                eprintln!("code lens · {capped} capped / {codeless} codeless");
                let c_q = embed_texts(
                    &client,
                    &cache,
                    &cli.code_model,
                    cli.dimensions,
                    std::slice::from_ref(&query),
                )
                .await?;
                let s = cosine_scores(&c_docs, &c_q[0]);
                fold_lens(
                    "<code>",
                    &s,
                    n,
                    combine,
                    &mut per_scope_detail,
                    &mut rank_lists,
                    &mut fused,
                );
            }

            let lens_count = rank_lists.len().max(1) as f32;
            match combine.as_str() {
                "rrf" => fused = rrf_fuse(&rank_lists, n),
                "mean" => fused.iter_mut().for_each(|f| *f /= lens_count),
                _ => {}
            }

            let order = argsort_desc(&fused);
            let results: Vec<_> = order
                .iter()
                .take(*top)
                .map(|&i| {
                    serde_json::json!({
                        "project": names[i],
                        "score": (fused[i] * 1e5).round() / 1e5,
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
                    println!("  {:.4}  {}", fused[i], names[i]);
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
