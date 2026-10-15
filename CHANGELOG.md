# Changelog

Earlier history has not been reconstructed here. The entries below identify
source changes without assigning package releases.

## Source history

- Reject duplicate or out-of-range embedding response indices before restoring input order.
  ([source change](https://github.com/arclabs561/scry/commit/0bd6b72863d0b9cdb3089be4ed4703f9d98db856))
- Represent absent code embeddings explicitly in `CodeEmbeddings`; missing code contributes no fusion vote, and code-only ranking and clustering preserve project identity.
  ([source change](https://github.com/arclabs561/scry/commit/d5911286d17eeba6eb204fc4ed8387ec96933153))
- Respect project `.gitignore` and `.ignore` rules for source embeddings by default; `--include-ignored` opts out while hard exclusions remain.
  ([source change](https://github.com/arclabs561/scry/commit/8876e17f7ce76902a1fcf2ee1a4134a3261a6888))
- Reject nonpositive requested embedding dimensions and cached or returned vectors whose width differs from an explicitly requested dimension.
  ([source change](https://github.com/arclabs561/scry/commit/708156ddf5720e4d41ceb39154819799ec34da79))
