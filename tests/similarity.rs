//! Integration tests for scry's public similarity API.
//!
//! scry's real document/query vectors come from an external OpenRouter
//! embedding model (`embed_texts`): async, network-bound, non-deterministic, and
//! requiring an API key. So these tests do NOT call the model. Instead they feed
//! hand-constructed embedding vectors through the deterministic, offline public
//! similarity layer (`cosine_scores`, `argsort_desc`, `l2_normalize`) and assert
//! both an unambiguous paraphrase-ranking outcome and the algebraic invariants of
//! cosine similarity. Every assertion calls a real exported `scry::` function;
//! none re-implements the logic under test.

/// Similarity of two single vectors via the public API. `cosine_scores`
/// normalizes both sides internally, so this is plain cosine similarity.
fn sim(a: &[f32], b: &[f32]) -> f32 {
    scry::cosine_scores(std::slice::from_ref(&a.to_vec()), b)[0]
}

// Hand-built 4-dim "embeddings". Two near-paraphrases (a fast in-memory
// key/value cache, phrased two ways) share most of their mass; the unrelated
// text (a watercolor painting tutorial) points elsewhere. The answer is
// unambiguous by construction: the paraphrases must be closer to each other than
// to the unrelated text.
const PARAPHRASE_A: [f32; 4] = [0.90, 0.10, 0.05, 0.00]; // "fast in-memory key-value cache"
const PARAPHRASE_B: [f32; 4] = [0.85, 0.15, 0.00, 0.05]; // "quick in-RAM key/value store"
const UNRELATED: [f32; 4] = [0.00, 0.05, 0.10, 0.95]; //   "watercolor landscape painting tutorial"

/// Invariant (the central one): the two paraphrases are mutually more similar
/// than either is to the unrelated text.
#[test]
fn paraphrases_are_closer_to_each_other_than_to_unrelated() {
    let ab = sim(&PARAPHRASE_A, &PARAPHRASE_B);
    let a_unrel = sim(&PARAPHRASE_A, &UNRELATED);
    let b_unrel = sim(&PARAPHRASE_B, &UNRELATED);

    assert!(
        ab > a_unrel && ab > b_unrel,
        "paraphrase pair similarity {ab} must exceed both cross similarities \
         ({a_unrel}, {b_unrel})"
    );
}

/// Invariant: ranking documents against a query orders the paraphrase ahead of
/// the unrelated document. Exercises `cosine_scores` + `argsort_desc` together,
/// the exact pair the `rank`/`ask` flow uses.
#[test]
fn ranking_puts_the_paraphrase_first() {
    // Query is paraphrase A; docs are [paraphrase B, unrelated]. B should win.
    let docs = vec![PARAPHRASE_B.to_vec(), UNRELATED.to_vec()];
    let scores = scry::cosine_scores(&docs, &PARAPHRASE_A);
    let order = scry::argsort_desc(&scores);
    assert_eq!(order, vec![0, 1], "scores were {scores:?}");
}

/// Invariant: cosine similarity is symmetric. sim(a, b) == sim(b, a).
#[test]
fn similarity_is_symmetric() {
    let ab = sim(&PARAPHRASE_A, &UNRELATED);
    let ba = sim(&UNRELATED, &PARAPHRASE_A);
    assert!((ab - ba).abs() < 1e-6, "symmetry violated: {ab} vs {ba}");
}

/// Invariant: a (non-zero) vector is maximally similar to itself, and 1.0 is the
/// ceiling no other comparison exceeds.
#[test]
fn self_similarity_is_the_maximum() {
    let self_sim = sim(&PARAPHRASE_A, &PARAPHRASE_A);
    assert!(
        (self_sim - 1.0).abs() < 1e-6,
        "self-similarity {self_sim} != 1.0"
    );

    let cross = sim(&PARAPHRASE_A, &PARAPHRASE_B);
    assert!(
        cross <= self_sim + 1e-6,
        "cross similarity {cross} should not exceed self similarity {self_sim}"
    );
}

/// Invariant: `cosine_scores` returns exactly one score per document, in
/// document order (shape contract the callers rely on).
#[test]
fn one_score_per_document_in_order() {
    let docs = vec![
        PARAPHRASE_A.to_vec(),
        PARAPHRASE_B.to_vec(),
        UNRELATED.to_vec(),
    ];
    let scores = scry::cosine_scores(&docs, &PARAPHRASE_A);
    assert_eq!(scores.len(), docs.len());
    // doc 0 is the query itself -> highest score of the three.
    let order = scry::argsort_desc(&scores);
    assert_eq!(order[0], 0, "self should rank first; scores {scores:?}");
}
