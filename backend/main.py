"""
Vanilla LLM Demo - Backend FastAPI
====================================

Backend stateless che espone un endpoint per chattare con un LLM "puro"
tramite Azure OpenAI. Nessun tool, nessuna ricerca web, nessuna memoria
lato server: tutto lo stato della conversazione vive nel browser e viene
rispedito ad ogni richiesta.

Scopo didattico: mostrare il comportamento di un Large Language Model
quando non è aiutato da nessuno strumento esterno.
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

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")  # password condivisa per la classe

# Limiti per turno (didattici, non tecnici)
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")  # per IP

# System prompt deliberatamente minimale - serve solo a evitare
# che il modello rifiuti di rispondere o si comporti in modo strano.
# NON contiene istruzioni "utili" che maschererebbero i limiti dell'LLM puro.
SYSTEM_PROMPT = (
    "Sei un modello linguistico. Rispondi alle domande dell'utente in modo "
    "diretto, in italiano, basandoti esclusivamente sulla tua conoscenza "
    "interna. Non hai accesso a internet, calcolatrici, o altri strumenti "
    "esterni. Se non sei sicuro di una risposta, prova comunque a rispondere "
    "secondo le tue informazioni."
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


# Client Azure OpenAI - inizializzato lazy alla prima richiesta
_client: Optional[AzureOpenAI] = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="Azure OpenAI non configurato. Mancano endpoint o API key.",
            )
        _client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _client


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


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, max_length=MAX_HISTORY_MESSAGES)


class ChatResponse(BaseModel):
    reply: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


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
    """Espone al frontend la configurazione 'visibile' (system prompt, modello).

    Trasparenza didattica: durante il corso si può mostrare letteralmente
    cosa c'è nel system prompt.
    """
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
    """Endpoint principale: riceve la cronologia, ritorna la prossima risposta.

    Stateless. Nessun tool. Nessuna ricerca web. Nessun reasoning aggiuntivo.
    Solo il modello che predice il prossimo token.
    """
    client = get_client()

    # Costruisci la lista di messaggi per OpenAI: system prompt + cronologia
    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in payload.messages:
        openai_messages.append({"role": msg.role, "content": msg.content})

    try:
        # Alcuni modelli (reasoning: o1, o3, o5, gpt-5) richiedono
        # `max_completion_tokens` invece di `max_tokens` e non accettano
        # `temperature` diversa dal default. Proviamo prima il dialetto
        # nuovo, ricadiamo sul vecchio se il modello lo supporta solo lui.
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=openai_messages,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                # Nessun tool. Nessuna funzione. Nessuna ricerca.
                tools=None,
            )
        except OpenAIError as new_dialect_error:
            err_str = str(new_dialect_error).lower()
            if "max_completion_tokens" in err_str or "unsupported" in err_str:
                # Modello classico (gpt-4o-mini, gpt-35-turbo, ecc.)
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
        raise HTTPException(
            status_code=502,
            detail=f"Errore dal modello: {exc}",
        ) from exc

    choice = response.choices[0]
    reply = choice.message.content or ""

    usage = response.usage
    return ChatResponse(
        reply=reply,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        model=AZURE_OPENAI_DEPLOYMENT,
    )


# ---------------------------------------------------------------------------
# Frontend statico
# ---------------------------------------------------------------------------

# Serve il file index.html dalla cartella ../frontend
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
