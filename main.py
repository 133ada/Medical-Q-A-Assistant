"""FastAPI application assembly for the medical knowledge assistant.

Only infrastructure initialization and router registration live here. HTTP
handlers are under ``App.Routers`` and business orchestration is under
``App.Services``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Allow ``python App/main.py`` to work when Python sets sys.path[0] to App/.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from App.LLM.llm import make_embedding_llm

from App.Integrations.langgraph import build_medical_graph
from App.Core.config import CONFIG_ALIYUN, FRONTEND_ORIGINS, missing_runtime_secrets
from App.Core.database import close_engine, init_db
from App.Integrations.memory import MemoryManager
from App.Repositories.conversation_summary_repository import create_short_term_store
from App.Integrations.rag import build_vectorstore_ddi, build_vectorstore_other
from App.Routers import legacy_router, router as api_router

VECTORSTORE_DDI_ADDITIONAL_QUANTITY = 0
VECTORSTORE_OTHER_ADDITIONAL_QUANTITY = 0


def _print_graph_structure(graph) -> None:
    """Print one helpful graph representation during application startup."""
    print("\nLangGraph structure:\n")
    try:
        print(graph.get_graph().draw_ascii())
    except Exception:
        try:
            print(graph.get_graph().draw_mermaid())
        except Exception as exc:
            print(f"Unable to print graph structure: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize external providers shared by the service layer."""
    print("\nInitializing medical knowledge assistant...\n")
    missing = missing_runtime_secrets()
    if missing:
        raise RuntimeError("Missing required runtime configuration: " + ", ".join(missing))

    embeddings_model = make_embedding_llm(
        model=CONFIG_ALIYUN["model"],
        openai_api_key=CONFIG_ALIYUN["api_key"],
        openai_api_base=CONFIG_ALIYUN["base_url"],
        dimensions=CONFIG_ALIYUN["dimensions"],
    )
    vectorstore_ddi = build_vectorstore_ddi(embeddings_model, VECTORSTORE_DDI_ADDITIONAL_QUANTITY)
    vectorstore_other = build_vectorstore_other(embeddings_model, VECTORSTORE_OTHER_ADDITIONAL_QUANTITY)
    print("Vector stores initialized")

    database_available = init_db()
    short_store = create_short_term_store()
    memory_manager = MemoryManager(short_store=short_store)
    print("Memory Services initialized")

    medical_graph = build_medical_graph(
        vectorstore_other, vectorstore_ddi, memory_manager=memory_manager
    )
    print("Medical graph initialized")
    _print_graph_structure(medical_graph)

    app.state.medical_graph = medical_graph
    app.state.memory_manager = memory_manager
    app.state.chat_reply_mode = os.environ.get("CHAT_REPLY_MODE", "medical_agent")
    app.state.health = {
        "graph": True,
        "vectorstore_other": vectorstore_other is not None,
        "vectorstore_ddi": vectorstore_ddi is not None,
        "database": database_available,
        "short_term_memory": short_store is not None,
    }
    print("Medical API started\n")
    try:
        yield
    finally:
        print("\nClosing medical API...\n")
        close_engine()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Medical Knowledge Assistant API",
        description="LangGraph medical RAG with optional authenticated Memory",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=FRONTEND_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    application.include_router(api_router)
    application.include_router(legacy_router)

    @application.get("/")
    async def root():
        return {
            "status": "running",
            "chat_endpoint": "POST /api/chat",
            "stream_endpoint": "POST /api/chat/stream",
            "session_endpoint": "POST /api/sessions/{session_id}/messages",
            "legacy_endpoints": ["POST /chat", "POST /chat/stream"],
        }

    @application.get("/health/live")
    async def health_live():
        return {"status": "ok"}

    @application.get("/health/ready")
    async def health_ready(request: Request):
        health = getattr(request.app.state, "health", {})
        ready = bool(health.get("graph") and health.get("vectorstore_other") and health.get("vectorstore_ddi"))
        payload = {"status": "ready" if ready else "not_ready", "dependencies": health}
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @application.get("/graph/viz")
    async def graph_viz(request: Request):
        graph = getattr(request.app.state, "medical_graph", None)
        if graph is None:
            return JSONResponse(status_code=503, content={"error": "Medical graph is not ready"})
        try:
            return {"format": "mermaid", "graph": graph.get_graph().draw_mermaid()}
        except Exception as exc:
            return {"error": str(exc), "hint": "draw_mermaid support is required"}

    return application


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
