from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.retriever import QueryRetriever, RetrievalError

router = APIRouter(prefix="/api", tags=["retrieval"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language query to retrieve relevant chunks for.")


class RetrievalItem(BaseModel):
    rank: int
    score: float
    document_id: str
    filename: str
    chunk_id: str
    page_number: int | None = None
    text: str


class QueryResponse(BaseModel):
    question: str
    results: list[RetrievalItem]
    total_results: int
    retrieval_time_ms: float


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    started_at = time.perf_counter()

    try:
        retriever = QueryRetriever()
        results = retriever.retrieve(question)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    retrieval_time_ms = (time.perf_counter() - started_at) * 1000
    response_results = [
        RetrievalItem(
            rank=item.rank,
            score=item.score,
            document_id=item.document_id,
            filename=item.filename,
            chunk_id=item.chunk_id,
            page_number=item.page_number,
            text=item.text,
        )
        for item in results
    ]

    return QueryResponse(
        question=question,
        results=response_results,
        total_results=len(response_results),
        retrieval_time_ms=retrieval_time_ms,
    )
