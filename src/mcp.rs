//! Minimal MCP server over stdio (newline-delimited JSON-RPC 2.0).
//!
//! Exposes scry's retrieval to an agent host (Claude Code, etc.) so it can call
//! `scry_query` / `scry_ask` directly instead of shelling out. The corpus is
//! whatever source the binary was launched with (local or `--github*`), built
//! once before the server loop. `scry_ask` is the full multi-step flow: a
//! planner LLM derives scope lenses, each retrieves, the rankings are
//! RRF-fused, and an LLM synthesizes an answer over the top hits.

use crate::cache::Cache;
use crate::corpus::{self, Project};
use crate::openrouter::Client;
use crate::{argsort_desc, cosine_scores, embed_texts};
use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const PLANNER: &str = "openai/gpt-4o-mini";

/// Run the stdio server loop until EOF.
pub async fn serve(
    projects: &[Project],
    client: &Client,
    cache: &Cache,
    model: &str,
    dims: Option<u32>,
) -> Result<()> {
    let names: Vec<String> = projects.iter().map(|p| p.name.clone()).collect();
    eprintln!("scry mcp: serving {} projects over stdio", projects.len());
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let result: Option<Value> = match method {
            "initialize" => Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "scry", "version": env!("CARGO_PKG_VERSION") }
            })),
            "tools/list" => Some(tools_list()),
            "tools/call" => {
                Some(handle_call(&req, projects, &names, client, cache, model, dims).await)
            }
            "ping" => Some(json!({})),
            _ => None, // notifications (no id) get no response
        };
        // Only requests (with an id) get a response; notifications do not.
        if let (Some(id), Some(result)) = (id, result) {
            let msg = json!({ "jsonrpc": "2.0", "id": id, "result": result });
            stdout.write_all(format!("{msg}\n").as_bytes()).await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

fn tools_list() -> Value {
    json!({ "tools": [
        {
            "name": "scry_query",
            "description": "Rank the corpus projects by relevance to a query (semantic search).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "the search query" },
                    "top": { "type": "integer", "description": "how many results (default 10)" }
                },
                "required": ["text"]
            }
        },
        {
            "name": "scry_ask",
            "description": "Answer a natural-language question about the corpus: a planner picks lenses, retrieval is RRF-fused, and an LLM synthesizes an answer citing projects.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": { "type": "string" },
                    "top": { "type": "integer", "description": "hits to synthesize over (default 8)" }
                },
                "required": ["question"]
            }
        }
    ]})
}

async fn handle_call(
    req: &Value,
    projects: &[Project],
    names: &[String],
    client: &Client,
    cache: &Cache,
    model: &str,
    dims: Option<u32>,
) -> Value {
    let params = req.get("params");
    let tool = params
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let args = params
        .and_then(|p| p.get("arguments"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let text = args
        .get("text")
        .or_else(|| args.get("question"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let top = args.get("top").and_then(Value::as_u64);

    let outcome = match tool {
        "scry_query" => {
            query_tool(
                text,
                top.unwrap_or(10) as usize,
                projects,
                names,
                client,
                cache,
                model,
                dims,
            )
            .await
        }
        "scry_ask" => {
            ask_tool(
                text,
                top.unwrap_or(8) as usize,
                projects,
                names,
                client,
                cache,
                model,
                dims,
            )
            .await
        }
        other => Err(anyhow::anyhow!("unknown tool {other:?}")),
    };
    match outcome {
        Ok(v) => json!({ "content": [{ "type": "text", "text": v.to_string() }] }),
        Err(e) => {
            json!({ "content": [{ "type": "text", "text": format!("error: {e}") }], "isError": true })
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn query_tool(
    text: &str,
    top: usize,
    projects: &[Project],
    names: &[String],
    client: &Client,
    cache: &Cache,
    model: &str,
    dims: Option<u32>,
) -> Result<Value> {
    let instr = corpus::resolve_scope("purpose");
    let inputs: Vec<String> = projects
        .iter()
        .map(|p| corpus::format_input(&instr, &p.readme))
        .collect();
    let docs = embed_texts(client, cache, model, dims, &inputs).await?;
    let qv = embed_texts(
        client,
        cache,
        model,
        dims,
        &[corpus::format_input(&instr, text)],
    )
    .await?;
    let scores = cosine_scores(&docs, &qv[0]);
    let order = argsort_desc(&scores);
    let results: Vec<Value> = order
        .iter()
        .take(top)
        .map(|&i| json!({ "project": names[i], "score": (scores[i] * 1e4).round() / 1e4 }))
        .collect();
    Ok(json!({ "results": results }))
}

#[allow(clippy::too_many_arguments)]
async fn ask_tool(
    text: &str,
    top: usize,
    projects: &[Project],
    _names: &[String],
    client: &Client,
    cache: &Cache,
    model: &str,
    dims: Option<u32>,
) -> Result<Value> {
    let res = crate::ask(
        client, cache, model, dims, PLANNER, projects, text, top, true,
    )
    .await?;
    let results: Vec<Value> = res
        .ranked
        .iter()
        .map(|(name, _)| json!({ "project": name }))
        .collect();
    Ok(json!({
        "answer": res.answer,
        "plan": { "scopes": res.scopes, "query": res.query },
        "results": results,
    }))
}
