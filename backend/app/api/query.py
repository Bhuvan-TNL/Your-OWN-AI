from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.llm import LLMConfigurationError, LLMProviderError
from app.services.rag_pipeline import RAGPipeline
from app.services.retriever import RetrievalError

router = APIRouter(prefix="/api", tags=["retrieval"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question for grounded retrieval and answer generation.")


class RetrievalItem(BaseModel):
    rank: int
    score: float
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    text: str


class SourceReference(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    rank: int | None = None
    score: float | None = None


class TimingInfo(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    results: list[RetrievalItem]
    timing: TimingInfo
    provider: str | None = None
    model: str | None = None
    # Legacy fields retained for existing clients while the structured timing
    # and source fields are adopted.
    total_results: int | None = None
    retrieval_time_ms: float | None = None


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        pipeline = RAGPipeline()
        response = pipeline.execute(question)
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM configuration error: {exc}",
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM provider unavailable: {exc}",
        ) from exc
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=[
            SourceReference(
                document_id=source.document_id,
                filename=source.filename,
                chunk_id=source.chunk_id,
                page_number=source.page_number,
                rank=source.rank,
                score=source.score,
            )
            for source in response.sources
        ],
        results=[
            RetrievalItem(
                rank=item["rank"],
                score=item["score"],
                document_id=item["document_id"],
                filename=item["filename"],
                chunk_id=item["chunk_id"],
                page_number=item.get("page_number"),
                text=item["text"],
            )
            for item in response.results
        ],
        timing=TimingInfo(**response.timing.to_dict()),
        provider=response.provider,
        model=response.model,
        total_results=len(response.results),
        retrieval_time_ms=response.timing.retrieval_ms,
    )
