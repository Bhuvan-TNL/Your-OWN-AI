from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.upload import router as upload_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("your_own_ai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting Your Own AI application")
    yield
    logger.info("Shutting down Your Own AI application")


app = FastAPI(
    title="Your Own AI",
    version="0.1.0",
    description="Local AI assistant foundation with a grounded knowledge base.",
    lifespan=lifespan,
)

app.include_router(upload_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "service": "your-own-ai",
        "version": app.version,
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
    }


@app.exception_handler(Exception)
async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )
