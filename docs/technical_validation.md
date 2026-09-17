# Phase 9 technical validation

## Scope and status

Phase 9 adds the Groq provider behind the existing provider-independent LLM
service and gates live evaluation behind an explicit smoke test. The Phase 7
retrieval artifacts remain unchanged. The optimized runtime retrieval
configuration is:

```text
CHUNK_SIZE=256
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.15
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Live Groq validation is currently **blocked** in this workspace because the
Groq key that was present during the audit was exposed in tool output and must
be revoked and replaced before any API call. No Groq request was made and no
real LLM answer metrics are reported. The blocked state is recorded in
`evaluation/results/groq_real_llm_baseline.json`.

## LLM integration

- Provider: `groq`, selected by `LLM_PROVIDER`.
- Model: `llama-3.3-70b-versatile`, selected by `LLM_MODEL`.
- Authentication: `GROQ_API_KEY` is loaded by the existing configuration layer
  from `backend/.env`; the Groq SDK receives it only when a live call is made.
- Request flow: `RAGPipeline` retrieves FAISS results, applies the configured
  similarity threshold, formats source/page/chunk context, and calls
  `LLMService`.
- Prompt construction: system instructions, retrieved context, question, and
  an instruction to answer only from context are assembled by `build_prompt`.
- Generation parameters: `max_tokens=256` and `temperature=0.2`.
- Groq request behavior: the official `groq` SDK sends separate system and user
  messages to `chat.completions.create`, with configurable model, temperature,
  and maximum tokens. The client uses a 90-second timeout.
- Error handling: missing keys, SDK/client failures, rate limits, timeouts,
  malformed responses, provider errors, empty responses, invalid configuration,
  and empty prompts are converted to typed application exceptions. The query
  API maps provider failures to HTTP 503 responses.
- Response parsing: the first chat choice's message content is returned;
  missing choices or empty content are rejected.

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

No real-provider result table is presented because the Groq key must be
rotated before live calls. The deterministic mock results in
`evaluation/results/experiment_summary.json` are retained only as Phase 7
retrieval-optimization evidence; they are not real LLM answer-quality results.

`evaluation/results/groq_vs_phase7_comparison.json` explicitly marks the real
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

- The configured `GROQ_API_KEY` was treated as compromised after exposure during
  the audit. It was not stored in source or result artifacts, and it must be
  revoked before live validation.
- The key is not stored in source, tests, JSON results, or documentation.
- `backend/.env` remains ignored by Git.
- Phase 7 result files, including `baseline.json` and
  `experiment_summary.json`, were not overwritten.

## Reproducibility

After revoking the exposed key and placing a replacement only in the local
ignored `backend/.env`, run:

```powershell
python backend/scripts/reindex_vector_store.py --apply
python evaluation/run_groq_validation.py --live
```

The runner performs a supported and unsupported smoke test before evaluating
all 19 questions. It writes the Groq result, source diagnostic, failure
analysis, and Phase 7 comparison under `evaluation/results/` without printing
the key. Run the backend regression suite afterward:

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
