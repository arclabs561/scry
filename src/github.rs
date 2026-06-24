//! GitHub corpus sources.
//!
//! Fetches repositories (a user's, an org's, a user's stars, or an explicit
//! list) and builds the same [`Project`] corpus the local scanner produces, so
//! all the embedding / retrieval machinery works unchanged. Each project's text
//! is the repo description plus topics (the dense signal) followed by the
//! README. There is no local source tree, so only the `readme` surface applies.

use crate::corpus::Project;
use anyhow::{Context, Result};
use base64::Engine;
use futures::stream::{self, StreamExt};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const README_CONCURRENCY: usize = 8;
const CHAR_BUDGET: usize = 24_000;

#[derive(Deserialize)]
struct Repo {
    name: String,
    full_name: String,
    description: Option<String>,
    #[serde(default)]
    fork: bool,
    #[serde(default)]
    topics: Vec<String>,
    #[serde(default)]
    archived: bool,
}

#[derive(Deserialize)]
struct ReadmeResp {
    content: String,
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .user_agent("scry")
        .build()
        .context("build github http client")
}

/// Auth token: `GITHUB_TOKEN` env, else `gh auth token` (the user's keyring
/// credential). Lifts the rate limit from 60/hr to 5000/hr.
fn token() -> Option<String> {
    if let Ok(t) = std::env::var("GITHUB_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    let out = std::process::Command::new("gh")
        .args(["auth", "token"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let t = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!t.is_empty()).then_some(t)
}

fn cache_path(key: &str) -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let safe: String = key
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    std::path::Path::new(&home).join(format!(".cache/scry/github/{safe}.json"))
}

fn read_cache(key: &str) -> Option<Vec<Project>> {
    let bytes = std::fs::read(cache_path(key)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn write_cache(key: &str, projects: &[Project]) {
    let cp = cache_path(key);
    if let Some(dir) = cp.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    if let Ok(bytes) = serde_json::to_vec(projects) {
        std::fs::write(&cp, bytes).ok();
    }
}

/// A user's owned repos. Forks/archived excluded unless `include_forks`.
pub async fn fetch_user(user: &str, include_forks: bool, refresh: bool) -> Result<Vec<Project>> {
    let key = format!("user-{user}");
    if !refresh {
        if let Some(p) = read_cache(&key) {
            eprintln!(
                "github: {} cached repos for {user} (--refresh to refetch)",
                p.len()
            );
            return Ok(p);
        }
    }
    let token = token();
    let http = client()?;
    let mut repos = paginate(
        &http,
        &format!("https://api.github.com/users/{user}/repos?type=owner&sort=updated"),
        token.as_deref(),
    )
    .await?;
    repos.retain(|r| !r.archived && (include_forks || !r.fork));
    finish(
        &key,
        repos,
        &http,
        token.as_deref(),
        &format!("user {user}"),
    )
    .await
}

/// An organization's repos. Forks/archived excluded unless `include_forks`.
pub async fn fetch_org(org: &str, include_forks: bool, refresh: bool) -> Result<Vec<Project>> {
    let key = format!("org-{org}");
    if !refresh {
        if let Some(p) = read_cache(&key) {
            eprintln!(
                "github: {} cached repos for org {org} (--refresh to refetch)",
                p.len()
            );
            return Ok(p);
        }
    }
    let token = token();
    let http = client()?;
    let mut repos = paginate(
        &http,
        &format!("https://api.github.com/orgs/{org}/repos?sort=updated"),
        token.as_deref(),
    )
    .await?;
    repos.retain(|r| !r.archived && (include_forks || !r.fork));
    finish(&key, repos, &http, token.as_deref(), &format!("org {org}")).await
}

/// A user's starred repos (all owners). Archived excluded; forks kept.
pub async fn fetch_stars(user: &str, refresh: bool) -> Result<Vec<Project>> {
    let key = format!("stars-{user}");
    if !refresh {
        if let Some(p) = read_cache(&key) {
            eprintln!(
                "github: {} cached stars for {user} (--refresh to refetch)",
                p.len()
            );
            return Ok(p);
        }
    }
    let token = token();
    let http = client()?;
    let mut repos = paginate(
        &http,
        &format!("https://api.github.com/users/{user}/starred"),
        token.as_deref(),
    )
    .await?;
    repos.retain(|r| !r.archived);
    finish(
        &key,
        repos,
        &http,
        token.as_deref(),
        &format!("stars of {user}"),
    )
    .await
}

/// An explicit `owner/name` list.
pub async fn fetch_named(repos: &[String], refresh: bool) -> Result<Vec<Project>> {
    let mut sorted = repos.to_vec();
    sorted.sort();
    let mut h = Sha256::new();
    h.update(sorted.join(",").as_bytes());
    let key = format!("repos-{:.12x}", h.finalize());
    if !refresh {
        if let Some(p) = read_cache(&key) {
            eprintln!(
                "github: {} cached repos from list (--refresh to refetch)",
                p.len()
            );
            return Ok(p);
        }
    }
    let token = token();
    let http = client()?;
    let fetched: Vec<Repo> = stream::iter(repos.iter())
        .map(|full| {
            let http = &http;
            let token = token.as_deref();
            async move {
                let url = format!("https://api.github.com/repos/{full}");
                let mut req = http
                    .get(&url)
                    .header("Accept", "application/vnd.github+json");
                if let Some(t) = token {
                    req = req.bearer_auth(t);
                }
                req.send()
                    .await
                    .ok()?
                    .error_for_status()
                    .ok()?
                    .json::<Repo>()
                    .await
                    .ok()
            }
        })
        .buffered(README_CONCURRENCY)
        .filter_map(|r| async move { r })
        .collect()
        .await;
    finish(&key, fetched, &http, token.as_deref(), "repo list").await
}

/// Page through a repo-list endpoint (100/page) until a short page.
async fn paginate(http: &reqwest::Client, base: &str, token: Option<&str>) -> Result<Vec<Repo>> {
    let mut out = Vec::new();
    let sep = if base.contains('?') { '&' } else { '?' };
    for page in 1..=10 {
        let url = format!("{base}{sep}per_page=100&page={page}");
        let mut req = http
            .get(&url)
            .header("Accept", "application/vnd.github+json");
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .context("list repos")?
            .error_for_status()
            .context("github repo list (rate limit? bad name?)")?;
        let batch: Vec<Repo> = resp.json().await.context("decode repo list")?;
        let n = batch.len();
        out.extend(batch);
        if n < 100 {
            break;
        }
    }
    Ok(out)
}

/// Fetch READMEs for `repos`, build projects, cache, and return.
async fn finish(
    key: &str,
    repos: Vec<Repo>,
    http: &reqwest::Client,
    token: Option<&str>,
    label: &str,
) -> Result<Vec<Project>> {
    if repos.is_empty() {
        anyhow::bail!("no repos found for {label} (private, empty, or all forks/archived)");
    }
    eprintln!(
        "github: {} repos for {label} (fetching READMEs)",
        repos.len()
    );
    let projects: Vec<Project> = stream::iter(repos.iter())
        .map(|r| async move {
            let readme = fetch_readme(http, &r.full_name, token)
                .await
                .unwrap_or_default();
            let mut meta: Vec<String> = Vec::new();
            if let Some(d) = &r.description {
                if !d.trim().is_empty() {
                    meta.push(d.trim().to_string());
                }
            }
            if !r.topics.is_empty() {
                meta.push(format!("topics: {}", r.topics.join(", ")));
            }
            let meta = meta.join(". ");
            let body = match (meta.is_empty(), readme.trim().is_empty()) {
                (true, true) => return None,
                (false, true) => meta,
                (true, false) => readme,
                (false, false) => format!("{meta}\n\n{readme}"),
            };
            let truncated: String = body.chars().take(CHAR_BUDGET).collect();
            Some(Project {
                name: r.name.clone(),
                dir: std::path::PathBuf::new(),
                readme: truncated,
            })
        })
        .buffered(README_CONCURRENCY)
        .filter_map(|p| async move { p })
        .collect()
        .await;
    write_cache(key, &projects);
    Ok(projects)
}

async fn fetch_readme(
    http: &reqwest::Client,
    full_name: &str,
    token: Option<&str>,
) -> Option<String> {
    let url = format!("https://api.github.com/repos/{full_name}/readme");
    let mut req = http
        .get(&url)
        .header("Accept", "application/vnd.github+json");
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let r: ReadmeResp = resp.json().await.ok()?;
    let raw = r.content.replace(['\n', '\r'], "");
    let bytes = base64::engine::general_purpose::STANDARD.decode(raw).ok()?;
    String::from_utf8(bytes).ok()
}
