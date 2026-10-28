# scry

Search and cluster a set of project READMEs.

`scry` builds embeddings for local project directories or public GitHub repos,
then ranks, clusters, or answers questions over that corpus. It can embed README
text, source files, or both. Embeddings are cached on disk so repeated queries do
not re-embed unchanged inputs.

The useful knob is the scope: an instruction such as "purpose", "techniques", or
"domain" is prepended before embedding, so the same projects can be ranked by
what they do, how they work, or the field they belong to.

## Install

```sh
cargo build --release
export PATH="$PWD/target/release:$PATH"
```

Set `OPENROUTER_API_KEY` in your environment before running queries.

## Corpus

By default the corpus is the local project dirs under `--root`. Point `--github
<user>` at any GitHub user to use their public repos instead (description +
topics + README per repo, `readme` surface only; there is no local tree for the
code surface). Forks and archived repos are excluded unless `--include-forks`.
The fetched corpus is cached per user under `~/.cache/scry/github/`; pass
`--refresh` to refetch. Auth comes from `GITHUB_TOKEN` or `gh auth token`
(5000/hr; unauthenticated 60/hr is fine for small users).

```sh
scry --github BurntSushi query "regular expression engine"
scry --github BurntSushi cluster --k 8
scry --github <user> ask "which repos do graph algorithms"
```

## Usage

```sh
# Cluster projects. Clusters are labeled by their medoid project.
scry cluster
scry cluster --scopes purpose --k 10
scry cluster --scopes "ml=the machine learning method used"

# Rank projects under one explicit scope.
scry query "approximate nearest neighbor search" \
    --scope "the search/indexing technique this project implements" --top 8
scry query "tools that talk to an LLM" --json

# Ask a question. A planner chooses scopes, retrieval is fused, and --answer
# synthesizes a cited reply over the top projects.
scry ask "which projects could help build code search" --top 6
scry ask "tools for ranking LLM outputs" --answer
scry --surface both ask "which projects implement vector search"

# Find near-duplicate or overlapping projects.
scry overlap --threshold 0.8

# Evaluate retrieval against a probes file: expected<TAB>query.
scry eval tests/probes.tsv          # top-1 / top-3 / MRR over labeled probes
scry eval tests/probes.tsv --answer # ask-answer quality: cited + grounded rates

# Run as an MCP stdio server.
scry mcp

# Embed source files instead of READMEs.
scry --surface code query "lock-free ring buffer"
scry --surface code cluster --k 10

# Use README and code together.
scry --surface both query "approximate nearest neighbor search"
scry --surface both cluster --scopes purpose --k 10
```

Shared flags (before the subcommand): `--root` (default: current directory),
`--model` (default `qwen/qwen3-embedding-8b`), `--dimensions` (Matryoshka
truncation), `--surface readme|code|both`, `--code-model` (default
`mistralai/codestral-embed-2505`), `--provider` (pin an OpenRouter provider for
deterministic embeddings; folded into the cache key).

## Surfaces

The `readme` surface leads each document with dense metadata (Cargo.toml /
package.json `description` + `keywords`) followed by the README, embedded under
an instruction scope. The `code` surface walks each project's source files and
embeds the most representative ones first: entry points (`lib.rs` / `main.rs` /
`mod.rs` / `__init__.py`), then anything under `src/`, skipping `benches/`,
`tests/`, `examples/`, `archive/`, `docs/`, and vendored/build trees (they
otherwise crowd out the real API within the per-project chunk cap). Chunks are
mean-pooled per project. Project-local `.gitignore` and `.ignore` rules apply by
default, including nested rules and negation. Use `--include-ignored` with code
or both surfaces to include ignored source; hard directory exclusions, source
symlink exclusion, and file/chunk limits still apply. Parent/global ignore
rules and `.git/info/exclude` are not loaded; local ignore files retain standard
semantics, including linked ignore configuration. iCloud-offloaded files are skipped. Projects with no
readable source retain their original identity but have no code representation;
this differs from a valid embedding whose coordinates happen to be zero.
Code-only query and ask results put missing rows last with a JSON `null` score
(`n/a` in text). Missing code supplies no vote to readme/code fusion. Code-only
clustering excludes these rows and maps them back to noise (`-1`); joint
readme/code clustering keeps their README with a zero code extension.

The `both` surface combines README and code: for `query` it Reciprocal-Rank-Fuses the
readme and code rankings, and for `cluster` it concatenates the L2-normalized
readme and code vectors into a joint feature space. Clustering uses a fixed seed.

## Multiple Scopes

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

## MCP

`scry mcp` serves the current corpus over stdio with `scry_query` and
`scry_ask` tools.

## Library API

Version 0.2 changes `embed_code` to return `CodeEmbeddings`, with one optional
vector per original project. `None` means no readable source; a present zero
vector remains valid. This source utility remains unpublished (`publish = false`).

## Cache

`~/.cache/scry/embeddings.json`, keyed by SHA-256 of `(model, provider, dimensions,
exact input text)`. The scope instruction is part of the input, so changing the lens
correctly produces new embeddings. Each run prints `cache: N hit / M miss`.

## License

MIT OR Apache-2.0
