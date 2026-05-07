"""
Script one-off per creare il vector store RAG su Azure OpenAI.

ATTENZIONE: questo script NON è idempotente — ogni esecuzione crea un NUOVO
vector store. Se devi rimpiazzare uno store esistente, cancellalo manualmente
dal portale Azure o via API prima di rieseguire.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
# Serve la preview che supporta Responses API + vector stores
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_RESPONSES", "2025-04-01-preview")

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
VECTOR_STORE_NAME = "baldan-knowledge-base"


def main() -> None:
    if not ENDPOINT or not API_KEY:
        print("❌ Mancano AZURE_OPENAI_ENDPOINT o AZURE_OPENAI_API_KEY nel .env", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Nessun PDF trovato in {KNOWLEDGE_BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"📂 Trovati {len(pdf_files)} PDF:")
    for f in pdf_files:
        print(f"   • {f.name}")

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    print(f"\n⏳ Creazione vector store '{VECTOR_STORE_NAME}'...")
    vs = client.vector_stores.create(name=VECTOR_STORE_NAME)
    print(f"   Store ID: {vs.id}")

    print("⏳ Upload e indicizzazione dei file (attendi il polling)...")
    file_streams = [open(f, "rb") for f in pdf_files]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs.id,
            files=file_streams,
        )
    finally:
        for s in file_streams:
            s.close()

    counts = batch.file_counts
    print(f"   Stato: {batch.status}")
    print(f"   File completati: {counts.completed}/{counts.total}  "
          f"(errori: {counts.failed}, in coda: {counts.in_progress})")

    if batch.status != "completed":
        print("⚠️  Il batch non ha completato correttamente — controlla i dettagli sopra.", file=sys.stderr)

    print(f"""
✅ Vector store creato: {vs.id}
Aggiungi al .env e alle App Settings di Azure:
    AZURE_VECTOR_STORE_ID={vs.id}
""")


if __name__ == "__main__":
    main()
