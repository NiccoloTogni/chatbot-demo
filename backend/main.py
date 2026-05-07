"""
Vanilla LLM Demo - Backend FastAPI
====================================

Backend stateless che espone un endpoint per chattare con un LLM "puro"
tramite Azure OpenAI. Nessun tool, nessuna ricerca web, nessuna memoria
lato server: tutto lo stato della conversazione vive nel browser e viene
rispedito ad ogni richiesta.

Scopo didattico: mostrare il comportamento di un Large Language Model
quando non è aiutato da nessuno strumento esterno (modalità Vanilla),
oppure quando ha accesso a una knowledge base aziendale (modalità RAG).
"""

import os
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI, OpenAIError
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

# Versione preview necessaria per Responses API + vector stores
AZURE_OPENAI_API_VERSION_RESPONSES = os.getenv(
    "AZURE_OPENAI_API_VERSION_RESPONSES", "2025-04-01-preview"
)
AZURE_VECTOR_STORE_ID = os.getenv("AZURE_VECTOR_STORE_ID", "")

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")  # password condivisa per la classe

# Limiti per turno (didattici, non tecnici)
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")  # per IP

SYSTEM_PROMPT = (
    "Sei un modello linguistico. Rispondi alle domande dell'utente in modo "
    "diretto, in italiano, basandoti esclusivamente sulla tua conoscenza "
    "interna. Non hai accesso a internet, calcolatrici, o altri strumenti "
    "esterni. Se non sei sicuro di una risposta, prova comunque a rispondere "
    "secondo le tue informazioni."
)

RAG_SYSTEM_PROMPT = (
    "Sei un assistente aziendale. Rispondi alle domande basandoti ESCLUSIVAMENTE "
    "sui documenti forniti tramite il tool file_search. Se l'informazione non è "
    "nei documenti, dillo esplicitamente. Cita sempre le fonti. Rispondi in italiano."
)

# ---------------------------------------------------------------------------
# Inizializzazione app
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Vanilla LLM Demo",
    description="Endpoint didattico per mostrare i limiti di un LLM senza tool",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dietro password, ok per uso didattico
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return HTTPException(
        status_code=429,
        detail="Troppe richieste. Aspetta qualche secondo e riprova.",
    )


# Client vanilla — api_version stabile
_client: Optional[AzureOpenAI] = None

# Client RAG — api_version preview per Responses API + vector stores
_rag_client: Optional[AzureOpenAI] = None


def _require_credentials() -> None:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Azure OpenAI non configurato. Mancano endpoint o API key.",
        )


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _require_credentials()
        _client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _client


def get_rag_client() -> AzureOpenAI:
    global _rag_client
    if _rag_client is None:
        _require_credentials()
        _rag_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION_RESPONSES,
        )
    return _rag_client


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

security = HTTPBearer(auto_error=False)


def verify_access(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    """Auth semplice: header `Authorization: Bearer <ACCESS_TOKEN>`.

    Se ACCESS_TOKEN non è settato in env, l'endpoint è aperto (utile in dev).
    """
    if not ACCESS_TOKEN:
        return
    if credentials is None or credentials.credentials != ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token mancante o non valido",
        )


# ---------------------------------------------------------------------------
# Modelli dati
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class Citation(BaseModel):
    file_name: str
    snippet: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, max_length=MAX_HISTORY_MESSAGES)
    mode: Literal["vanilla", "rag"] = "vanilla"


class ChatResponse(BaseModel):
    reply: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    citations: Optional[List[Citation]] = None


class ConfigResponse(BaseModel):
    """Configurazione esposta al frontend (no segreti)."""

    model_deployment: str
    system_prompt: str
    max_output_tokens: int
    max_history_messages: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """Endpoint di salute - usato dagli orchestratori (k8s, ACA)."""
    return {"status": "ok"}


@app.get("/api/config", response_model=ConfigResponse)
def get_config(_: None = Depends(verify_access)) -> ConfigResponse:
    return ConfigResponse(
        model_deployment=AZURE_OPENAI_DEPLOYMENT,
        system_prompt=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_history_messages=MAX_HISTORY_MESSAGES,
    )


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT)
def chat(
    request: Request,  # richiesto da slowapi
    payload: ChatRequest,
    _: None = Depends(verify_access),
) -> ChatResponse:
    if payload.mode == "rag":
        return _chat_rag(payload)
    return _chat_vanilla(payload)


def _chat_vanilla(payload: ChatRequest) -> ChatResponse:
    """Flusso vanilla: chat completions senza tool."""
    client = get_client()

    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in payload.messages:
        openai_messages.append({"role": msg.role, "content": msg.content})

    try:
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=openai_messages,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                tools=None,
            )
        except OpenAIError as new_dialect_error:
            err_str = str(new_dialect_error).lower()
            if "max_completion_tokens" in err_str or "unsupported" in err_str:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=openai_messages,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.7,
                    tools=None,
                )
            else:
                raise
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Errore dal modello: {exc}") from exc

    choice = response.choices[0]
    usage = response.usage
    return ChatResponse(
        reply=choice.message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        model=AZURE_OPENAI_DEPLOYMENT,
    )


def _chat_rag(payload: ChatRequest) -> ChatResponse:
    """Flusso RAG: Responses API con file_search su vector store Azure."""
    if not AZURE_VECTOR_STORE_ID:
        raise HTTPException(
            status_code=500,
            detail="AZURE_VECTOR_STORE_ID non configurato. Esegui prima scripts/setup_vector_store.py.",
        )

    client = get_rag_client()

    input_messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    try:
        response = client.responses.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            instructions=RAG_SYSTEM_PROMPT,
            input=input_messages,
            tools=[{"type": "file_search", "vector_store_ids": [AZURE_VECTOR_STORE_ID]}],
            include=["file_search_call.results"],
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Errore dal modello RAG: {exc}") from exc

    # Raccoglie snippet dai risultati del file_search (disponibili grazie a include=...)
    snippets_by_file_id: dict[str, str] = {}
    for item in response.output:
        if item.type == "file_search_call" and item.results:
            for result in item.results:
                if result.file_id and result.text and result.file_id not in snippets_by_file_id:
                    snippets_by_file_id[result.file_id] = result.text

    # Estrae testo e citazioni dalle annotation del messaggio di output
    reply_text = ""
    citations: List[Citation] = []
    seen_file_ids: set[str] = set()

    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    reply_text += block.text
                    for ann in block.annotations:
                        if ann.type == "file_citation" and ann.file_id not in seen_file_ids:
                            seen_file_ids.add(ann.file_id)
                            display_name = _format_filename(ann.filename or ann.file_id)
                            snippet = snippets_by_file_id.get(ann.file_id)
                            citations.append(Citation(file_name=display_name, snippet=snippet))

    usage = response.usage
    return ChatResponse(
        reply=reply_text,
        prompt_tokens=usage.input_tokens if usage else 0,
        completion_tokens=usage.output_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        model=AZURE_OPENAI_DEPLOYMENT,
        citations=citations if citations else None,
    )


def _format_filename(raw: str) -> str:
    """Converte 'PQ-04_NonConformita_Fornitori.pdf' → 'PQ-04 NonConformita Fornitori'."""
    name = raw.removesuffix(".pdf")
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Frontend statico
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )

    @app.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
