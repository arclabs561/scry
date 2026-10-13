//! Minimal OpenRouter client.
//!
//! Only the two endpoints this tool needs: `/embeddings` (OpenAI-compatible) and
//! `/chat/completions` (for the dual-LLM planner). The API key is read once from
//! the `OPENROUTER_API_KEY` environment variable.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

const EMBED_URL: &str = "https://openrouter.ai/api/v1/embeddings";
const CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

pub(crate) fn requested_width(dimensions: Option<u32>) -> Result<Option<usize>> {
    anyhow::ensure!(
        dimensions != Some(0),
        "dimensions must be greater than zero"
    );
    Ok(dimensions.map(|d| d as usize))
}

pub(crate) fn check_width(vector: &[f32], expected: Option<usize>, row: usize) -> Result<()> {
    if let Some(expected) = expected {
        anyhow::ensure!(
            vector.len() == expected,
            "embedding width mismatch at row {row}: expected {expected}, got {}",
            vector.len()
        );
    }
    Ok(())
}

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
    fn into_embeddings(
        mut self,
        input_count: usize,
        dimensions: Option<u32>,
    ) -> Result<Vec<Vec<f32>>> {
        let width = requested_width(dimensions)?;
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
            check_width(&item.embedding, width, expected)?;
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
    /// An explicit dimension must be positive and match every returned vector.
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
        requested_width(dimensions)?;
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
                    return parsed.into_embeddings(inputs.len(), dimensions);
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
mod dimension_tests {
    use super::{Client, EmbedResp};
    use crate::{cache::Cache, cosine_scores, embed_texts};

    fn client() -> Client {
        Client {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_millis(50))
                .build()
                .unwrap(),
            key: String::new(),
            provider: None,
        }
    }

    fn cache() -> Cache {
        Cache::load(cache_path())
    }

    fn cache_path() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "scry-width-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    #[test]
    fn explicit_provider_width_rejects_short_long_and_empty_rows() {
        for row in [vec![], vec![1.0], vec![1.0, 0.0, 0.0]] {
            let response: EmbedResp = serde_json::from_value(serde_json::json!({
                "data": [{"index": 0, "embedding": row}]
            }))
            .unwrap();
            let error = response.into_embeddings(1, Some(2)).unwrap_err();
            assert!(error.to_string().contains("width mismatch"), "{error}");
        }
    }

    #[test]
    fn provider_width_preserves_order_zero_vectors_and_default_width() {
        let empty: EmbedResp = serde_json::from_str(r#"{"data":[]}"#).unwrap();
        assert!(empty.into_embeddings(0, Some(2)).unwrap().is_empty());
        let data =
            r#"{"data":[{"index":1,"embedding":[0.0,0.0]},{"index":0,"embedding":[1.0,0.0]}]}"#;
        let response: EmbedResp = serde_json::from_str(data).unwrap();
        assert_eq!(
            response.into_embeddings(2, Some(2)).unwrap(),
            [vec![1.0, 0.0], vec![0.0, 0.0]]
        );
        let response: EmbedResp = serde_json::from_str(
            r#"{"data":[{"index":0,"embedding":[]},{"index":1,"embedding":[1.0]}]}"#,
        )
        .unwrap();
        assert_eq!(
            response.into_embeddings(2, None).unwrap(),
            [vec![], vec![1.0]]
        );
    }

    #[tokio::test]
    async fn explicit_cached_width_rejects_short_long_and_empty_rows() {
        let cache = cache();
        for row in [vec![], vec![1.0], vec![1.0, 0.0, 0.0]] {
            cache.insert(Cache::key("model", None, Some(2), "doc"), row);
            let error = embed_texts(&client(), &cache, "model", Some(2), &["doc".into()])
                .await
                .unwrap_err();
            assert!(error.to_string().contains("width mismatch"), "{error}");
        }
    }

    #[tokio::test]
    async fn invalid_cache_hit_rejects_before_requesting_missing_input() {
        let cache = cache();
        cache.insert(Cache::key("model", None, Some(2), "doc"), vec![1.0]);
        let error = embed_texts(
            &client(),
            &cache,
            "model",
            Some(2),
            &["missing".into(), "doc".into()],
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("width mismatch"), "{error}");
        assert!(!cache.contains_each(&[Cache::key("model", None, Some(2), "missing")])[0]);
    }

    #[tokio::test]
    async fn zero_dimensions_fail_locally_even_for_empty_input() {
        let cache = cache();
        let error = embed_texts(&client(), &cache, "model", Some(0), &[])
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dimensions must be greater than zero"),
            "{error}"
        );
        let error = client().embed("model", Some(0), &[]).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("dimensions must be greater than zero"),
            "{error}"
        );
        assert!(embed_texts(&client(), &cache, "model", Some(2), &[])
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn direct_provider_zero_dimensions_fail_before_transport() {
        let error = client()
            .embed("model", Some(0), &["doc".into()])
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "dimensions must be greater than zero");
    }

    #[tokio::test]
    async fn persisted_invalid_width_is_rejected_without_rewriting_cache() {
        let path = cache_path();
        let cache = Cache::load(path.clone());
        cache.insert(Cache::key("model", None, Some(2), "doc"), vec![1.0]);
        cache.save().unwrap();
        let original = std::fs::read(&path).unwrap();
        let reloaded = Cache::load(path.clone());
        let result = embed_texts(&client(), &reloaded, "model", Some(2), &["doc".into()]).await;
        let after = std::fs::read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            result.unwrap_err().to_string(),
            "embedding width mismatch at row 0: expected 2, got 1"
        );
        assert_eq!(original, after);
    }

    #[test]
    fn final_cache_snapshot_rejects_replacement_after_preflight() {
        let cache = cache();
        let key = Cache::key("model", None, Some(2), "doc");
        cache.insert(key.clone(), vec![1.0, 0.0]);
        super::check_width(&cache.get_each(std::slice::from_ref(&key))[0], Some(2), 0).unwrap();
        // Model a replacement during the producer's await, then call the same
        // final cache-read boundary used by embed_texts.
        cache.insert(key.clone(), vec![1.0]);
        let error = crate::cached_embeddings(&cache, &[key], Some(2)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "embedding width mismatch at row 0: expected 2, got 1"
        );
    }

    #[tokio::test]
    async fn valid_cached_width_preserves_zero_vectors_and_scoring() {
        let cache = cache();
        for (text, row) in [
            ("doc", vec![1.0, 0.0]),
            ("zero", vec![0.0, 0.0]),
            ("query", vec![1.0, 0.0]),
        ] {
            cache.insert(Cache::key("model", None, Some(2), text), row);
        }
        let vectors = embed_texts(
            &client(),
            &cache,
            "model",
            Some(2),
            &["doc".into(), "zero".into(), "query".into()],
        )
        .await
        .unwrap();
        assert_eq!(vectors, [vec![1.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]]);
        assert_eq!(cosine_scores(&vectors[..2], &vectors[2]), [1.0, 0.0]);
        assert_eq!(cache.hits(), 3);
        assert_eq!(cache.misses(), 0);
    }

    #[tokio::test]
    async fn default_width_and_cache_key_policy_are_unchanged() {
        let cache = cache();
        cache.insert(Cache::key("model", None, None, "doc"), vec![]);
        assert_eq!(
            embed_texts(&client(), &cache, "model", None, &["doc".into()])
                .await
                .unwrap(),
            [Vec::<f32>::new()]
        );
        assert_ne!(
            Cache::key("model", Some("a"), Some(2), "doc"),
            Cache::key("model", Some("b"), Some(2), "doc")
        );
        assert_ne!(
            Cache::key("model", None, Some(1), "doc"),
            Cache::key("model", None, Some(2), "doc")
        );
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
        assert!(response.into_embeddings(2, None).is_err());
    }

    #[test]
    fn embeddings_reject_out_of_range_indices() {
        let response: EmbedResp = serde_json::from_str(
            r#"{"data":[{"index":0,"embedding":[1.0]},{"index":2,"embedding":[2.0]}]}"#,
        )
        .unwrap();
        assert!(response.into_embeddings(2, None).is_err());
    }

    #[test]
    fn embeddings_restore_input_order() {
        let response: EmbedResp = serde_json::from_str(
            r#"{"data":[{"index":1,"embedding":[2.0]},{"index":0,"embedding":[1.0]}]}"#,
        )
        .unwrap();
        assert_eq!(
            response.into_embeddings(2, None).unwrap(),
            vec![vec![1.0], vec![2.0]]
        );
    }

    #[test]
    fn embeddings_preserve_count_and_empty_handling() {
        let empty: EmbedResp = serde_json::from_str(r#"{"data":[]}"#).unwrap();
        assert!(empty.into_embeddings(0, None).unwrap().is_empty());
        let missing: EmbedResp = serde_json::from_str(r#"{"data":[]}"#).unwrap();
        assert!(missing
            .into_embeddings(1, None)
            .unwrap_err()
            .to_string()
            .contains("count mismatch"));
    }
}
