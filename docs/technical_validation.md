# Phase 8 technical validation

## Scope and status

Phase 8 validates the existing retrieval-augmented generation pipeline with the
configured Hugging Face provider. The Phase 7 retrieval artifacts remain
unchanged. The optimized runtime retrieval configuration is:

```text
CHUNK_SIZE=256
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.15
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Live validation is currently **blocked** because `HF_TOKEN` is not configured
in the local `backend/.env`. No Hugging Face request was made and no real LLM
answer metrics are reported. The blocked state is recorded in
`evaluation/results/real_llm_baseline.json`.

## Existing LLM integration

- Provider: `huggingface`, selected by `LLM_PROVIDER`.
- Model: `google/flan-t5-base`, selected by `LLM_MODEL`.
- Authentication: `HF_TOKEN` is loaded by the existing configuration layer
  from `backend/.env`; the provider sends it as an Authorization Bearer header
  only when a live call is made.
- Request flow: `RAGPipeline` retrieves FAISS results, applies the configured
  similarity threshold, formats source/page/chunk context, and calls
  `LLMService`.
- Prompt construction: system instructions, retrieved context, question, and
  an instruction to answer only from context are assembled by `build_prompt`.
- Generation parameters: `max_new_tokens=256`, `temperature=0.2`, and
  `return_full_text=false`.
- HTTP behavior: `httpx.Client` posts to the Hugging Face inference endpoint
  with a 90-second timeout.
- Error handling: HTTP failures, invalid JSON, provider error payloads, empty
  responses, invalid configuration, and empty prompts are converted to typed
  configuration/provider errors. The query API maps provider failures to HTTP
  503 responses.
- Response parsing: list responses use `generated_text`; dictionary responses
  support `generated_text` or provider error payloads. Empty generated text is
  rejected.

## Index validation

The safe reindexing script was run with `--apply` after applying the Phase 7
configuration. The generated index contains:

- 2 deterministic document IDs;
- 211 chunks;
- 2 unique original-content hashes;
- 0 duplicate content-hash/chunk keys;
- page metadata on every chunk;
- a rebuilt FAISS index.

Raw PDF/TXT files were not modified or deleted.

## Evaluation methodology

The existing 19-question dataset was used without modification:

- 11 supported factual questions;
- 3 conceptual questions;
- 1 comparison question;
- 3 multi-sentence questions;
- 1 unsupported question.

The real evaluation runner records the question, reference answer, generated
answer, retrieved source metadata, pages, scores, provider/model, timing,
refusal status, and existing evaluation metrics. It separates retrieval metrics
from generation/answer-quality metrics and writes a separate source/citation
diagnostic.

Required metrics are filename hit@k, page hit@k, supported token F1, supported
embedding cosine, groundedness token overlap, unsupported refusal rate, mean
retrieval/generation/total latency, and total p95 latency.

## Results

No real-provider result table is presented because the credential precondition
was not satisfied. The deterministic mock results in
`evaluation/results/experiment_summary.json` are retained only as Phase 7
retrieval-optimization evidence; they are not real LLM answer-quality results.

`evaluation/results/real_vs_mock_comparison.json` explicitly marks the real
side as unavailable rather than comparing incomparable measurements.

## Source correctness and failure analysis

`evaluation/results/source_citation_correctness.json` is a blocked diagnostic.
It does not infer citation correctness without generated answers. Likewise,
`evaluation/results/failure_analysis.json` contains no fabricated real-provider
examples. Once live validation is available, failures will be classified using
observed evidence such as retrieval misses, page mismatches, weak answer overlap,
provider errors, and unsupported-question behavior. Token overlap is only a
diagnostic; it does not prove semantic entailment or absence of unsupported
claims.

## Security checks

- `HF_TOKEN` was checked only for presence; its value was never printed.
- The token is not stored in source, tests, JSON results, documentation, or
  logs.
- `backend/.env` remains ignored by Git.
- Phase 7 result files, including `baseline.json` and
  `experiment_summary.json`, were not overwritten.

## Reproducibility

After placing a valid token in the local ignored `backend/.env`, run:

```powershell
python backend/scripts/reindex_vector_store.py --apply
python evaluation/run_phase8_validation.py
```

The runner performs a supported and unsupported smoke test before evaluating
all 19 questions. It writes the real result, source diagnostic, failure
analysis, and mock-vs-real comparison under `evaluation/results/` without
printing the token. Run the backend regression suite afterward:

```powershell
cd backend
python -m pytest -q
```

## Limitations

The dataset is small and contains two unique documents. Page metadata denotes
the first page associated with a chunk, and timing measurements are not a
statistical performance study. Human review or a semantic entailment evaluator
is still required for strong claims about citation correctness and unsupported
claims.
