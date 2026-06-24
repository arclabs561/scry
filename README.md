# scry

Scry your project corpus. Embeds each project's README under one or more *scopes*
(instruction lenses) via OpenRouter, then clusters them, ranks them against a
query, or answers a natural-language question by letting a planner LLM choose the
lenses. Embeddings are cached on disk so unchanged inputs are never re-embedded.

The distinguishing idea is the scope: an instruction (purpose / techniques /
domain, or any free text) prepended to each document before embedding, so the
same corpus ranks and groups differently depending on the lens.

## Build

```sh
cargo build --release        # binary at target/release/scry
export OPENROUTER_API_KEY=... # lives in ~/.localrc
```

Clustering reuses the [`clump`](https://github.com/arclabs561/clump) crate (EVoC);
it is an unpublished sibling, so the dependency is a `path` and this crate is
local-only.

## Corpus source

By default the corpus is the local project dirs under `--root`. Point `--github
<user>` at any GitHub user to use their public repos instead (description +
topics + README per repo, `readme` surface only — there is no local tree for the
code surface). Forks and archived repos are excluded unless `--include-forks`.
The fetched corpus is cached per user under `~/.cache/scry/github/`; pass
`--refresh` to refetch. Auth comes from `GITHUB_TOKEN` or `gh auth token`
(5000/hr; unauthenticated 60/hr is fine for small users).

```sh
scry --github BurntSushi query "regular expression engine"
scry --github BurntSushi cluster --k 8
scry --github <user> ask "which repos do graph algorithms"
```

## Subcommands

```sh
# cluster (EVoC; clusters labeled by their medoid project)
scry cluster
scry cluster --scopes purpose --k 10
scry cluster --scopes "ml=the machine learning method used"

# query: rank projects under one explicit lens
scry query "approximate nearest neighbor search" \
    --scope "the search/indexing technique this project implements" --top 8
scry query "tools that talk to an LLM" --json

# ask: plan scope(s) -> retrieve -> RRF-fuse; --answer synthesizes a cited reply
scry ask "which projects could help build code search" --top 6
scry ask "tools for ranking LLM outputs" --answer
scry --surface both ask "which projects implement vector search"  # +code lens

# overlap: near-duplicate / overlapping projects (high pairwise cosine)
scry overlap --threshold 0.8

# eval: retrieval accuracy against a probes file (expected<TAB>query lines)
scry eval tests/probes.tsv          # top-1 / top-3 / MRR over labeled probes
scry eval tests/probes.tsv --answer # ask-answer quality: cited + grounded rates

# mcp: run as an MCP stdio server so an agent host calls scry_query / scry_ask
scry mcp
scry --github BurntSushi mcp        # serve a remote corpus

# code surface: embed source files instead of READMEs (for cluster + query)
scry --surface code query "lock-free ring buffer"
scry --surface code cluster --k 10

# both: fuse readme + code (query = RRF of the two rankings; cluster = joint features)
scry --surface both query "approximate nearest neighbor search"
scry --surface both cluster --scopes purpose --k 10
```

Shared flags (before the subcommand): `--root` (default `~/Documents/dev`),
`--model` (default `qwen/qwen3-embedding-8b`), `--dimensions` (Matryoshka
truncation), `--surface readme|code|both`, `--code-model` (default
`mistralai/codestral-embed-2505`), `--provider` (pin an OpenRouter provider for
deterministic embeddings; folded into the cache key).

## What gets embedded

The `readme` surface leads each document with dense metadata (Cargo.toml /
package.json `description` + `keywords`) followed by the README, embedded under
an instruction scope. The `code` surface walks each project's source files and
embeds the most representative ones first: entry points (`lib.rs` / `main.rs` /
`mod.rs` / `__init__.py`), then anything under `src/`, skipping `benches/`,
`tests/`, `examples/`, `archive/`, `docs/`, and vendored/build trees (they
otherwise crowd out the real API within the per-project chunk cap). Chunks are
mean-pooled to one vector per project. iCloud-offloaded files are skipped.

## Surfaces

The `both` surface combines them: for `query` it Reciprocal-Rank-Fuses the
readme and code rankings, and for `cluster` it concatenates the L2-normalized
readme and code vectors into a joint feature space. All three subcommands honor
`--surface`: for `ask`, the code ranking
joins the readme scope rankings as one more lens in the fusion (and with
`--surface code`, the code ranking is the only lens; with `--surface both`, both;
clustering is deterministic via a fixed seed).

## Combining multiple scopes

`ask` lets the planner pick 1-3 lenses. Cosine ranges differ per lens, so raw
scores are not comparable across scopes; fusion defaults to Reciprocal Rank
Fusion (`--combine rrf`), which combines per-lens *ranks* and is scale-free.
`mean` and `max` are available. The JSON output always returns each project's
per-scope `rank` and `cosine`, so a consuming agent sees which lens drove a hit:

```json
{"question":"...","plan":{"scopes":["..."],"query":"..."},"combine":"rrf",
 "results":[{"project":"rankit","score":0.0333,
             "scopes":[{"scope":"...","rank":0,"cosine":0.66}]}]}
```

## Models (all on OpenRouter)

- Default embeddings: `qwen/qwen3-embedding-8b` (4096-dim, 32k context).
- Code-aware embeddings: `mistralai/codestral-embed-2505`, `google/gemini-embedding-2`.
- Planner (`ask`): any chat model via `--planner` (default `openai/gpt-4o-mini`).

## Cache

`~/.cache/scry/embeddings.json`, keyed by SHA-256 of `(model, dimensions, exact
input text)`. The scope instruction is part of the input, so changing the lens
correctly produces new embeddings. Each run prints `cache: N hit / M miss`.
