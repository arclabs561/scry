//! Minimal OpenRouter client.
//!
//! Only the two endpoints this tool needs: `/embeddings` (OpenAI-compatible) and
//! `/chat/completions` (for the dual-LLM planner). The API key is read once from
//! the `OPENROUTER_API_KEY` environment variable.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

const EMBED_URL: &str = "https://openrouter.ai/api/v1/embeddings";
const CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// A thin authenticated wrapper around `reqwest::Client`.
pub struct Client {
    http: reqwest::Client,
    key: String,
    provider: Option<String>,
}

#[derive(Deserialize)]
struct EmbedResp {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
    index: usize,
}

impl EmbedResp {
    fn into_embeddings(mut self, input_count: usize) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            self.data.len() == input_count,
            "embeddings count mismatch: got {} for {} inputs",
            self.data.len(),
            input_count
        );
        self.data.sort_by_key(|d| d.index);
        for (expected, item) in self.data.iter().enumerate() {
            anyhow::ensure!(
                item.index == expected,
                "embeddings index mismatch: expected {}, got {}",
                expected,
                item.index
            );
        }
        Ok(self.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[derive(Deserialize)]
struct ChatResp {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: String,
}

impl Client {
    /// Build a client, reading `OPENROUTER_API_KEY` from the environment.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .connect_timeout(std::time::Duration::from_secs(20))
            .build()
            .context("build http client")?;
        Ok(Self {
            http,
            key,
            provider: None,
        })
    }

    /// Pin a specific upstream provider (OpenRouter `provider.only`). Makes
    /// embeddings deterministic across runs and is folded into the cache key, so
    /// vectors from different providers never mix in the cache.
    pub fn with_provider(mut self, provider: Option<String>) -> Self {
        self.provider = provider;
        self
    }

    /// The pinned provider, if any (part of the cache key).
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }

    /// Embed a batch of inputs. Returns one vector per input, in input order.
    ///
    /// Retries transient failures with exponential backoff. The embedding model
    /// must support batched `input` arrays (Qwen3-Embedding and the OpenAI
    /// models on OpenRouter do).
    pub async fn embed(
        &self,
        model: &str,
        dimensions: Option<u32>,
        inputs: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let mut body = serde_json::json!({ "model": model, "input": inputs });
        if let Some(d) = dimensions {
            body["dimensions"] = d.into();
        }
        if let Some(p) = &self.provider {
            body["provider"] = serde_json::json!({ "only": [p] });
        }
        for attempt in 0..4u32 {
            let resp = self
                .http
                .post(EMBED_URL)
                .bearer_auth(&self.key)
                .header("X-Title", "scry")
                .json(&body)
                .send()
                .await
                .and_then(|r| r.error_for_status());
            match resp {
                Ok(r) => {
                    let parsed: EmbedResp = r.json().await.context("decode embeddings")?;
                    return parsed.into_embeddings(inputs.len());
                }
                Err(e) if attempt == 3 => return Err(e).context("embeddings request failed"),
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                }
            }
        }
        unreachable!()
    }

    /// Single chat completion constrained to a JSON object. Returns the raw
    /// message content (the caller parses it).
    pub async fn chat_json(&self, model: &str, system: &str, user: &str) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        });
        let resp: ChatResp = self
            .http
            .post(CHAT_URL)
            .bearer_auth(&self.key)
            .header("X-Title", "scry")
            .json(&body)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .context("chat request failed")?
            .json()
            .await
            .context("decode chat response")?;
        resp.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow!("chat response had no choices"))
    }

    /// Free-form chat completion (no JSON constraint). Returns message content.
    pub async fn chat(&self, model: &str, system: &str, user: &str) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        });
        let resp: ChatResp = self
            .http
            .post(CHAT_URL)
            .bearer_auth(&self.key)
            .header("X-Title", "scry")
            .json(&body)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .context("chat request failed")?
            .json()
            .await
            .context("decode chat response")?;
        resp.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow!("chat response had no choices"))
    }
}

#[cfg(test)]
mod tests {
    use super::EmbedResp;

    #[test]
    fn embeddings_reject_duplicate_indices() {
        let response: EmbedResp = serde_json::from_str(
            r#"{"data":[{"index":0,"embedding":[1.0]},{"index":0,"embedding":[2.0]}]}"#,
        )
        .unwrap();
        assert!(response.into_embeddings(2).is_err());
    }

    #[test]
    fn embeddings_reject_out_of_range_indices() {
        let response: EmbedResp = serde_json::from_str(
            r#"{"data":[{"index":0,"embedding":[1.0]},{"index":2,"embedding":[2.0]}]}"#,
        )
        .unwrap();
        assert!(response.into_embeddings(2).is_err());
    }

    #[test]
    fn embeddings_restore_input_order() {
        let response: EmbedResp = serde_json::from_str(
            r#"{"data":[{"index":1,"embedding":[2.0]},{"index":0,"embedding":[1.0]}]}"#,
        )
        .unwrap();
        assert_eq!(
            response.into_embeddings(2).unwrap(),
            vec![vec![1.0], vec![2.0]]
        );
    }

    #[test]
    fn embeddings_preserve_count_and_empty_handling() {
        let empty: EmbedResp = serde_json::from_str(r#"{"data":[]}"#).unwrap();
        assert!(empty.into_embeddings(0).unwrap().is_empty());
        let missing: EmbedResp = serde_json::from_str(r#"{"data":[]}"#).unwrap();
        assert!(missing
            .into_embeddings(1)
            .unwrap_err()
            .to_string()
            .contains("count mismatch"));
    }
}
