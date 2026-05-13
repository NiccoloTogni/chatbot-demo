"""
Script one-off per creare il vector store rossi-preventivi-kb su Azure OpenAI.

ATTENZIONE: ogni esecuzione crea un NUOVO vector store. Vecchi da eliminare
manualmente dal portale Azure prima di rieseguire.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_RESPONSES", "2025-04-01-preview")

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
VECTOR_STORE_NAME = "rossi-preventivi-kb"
TARGET_FILES = {"ST-MAT-01_Materiali_Approvati.pdf", "IO-PROD-07_Stampaggio_Iniezione.pdf"}


def main() -> None:
    if not ENDPOINT or not API_KEY:
        print("❌ Mancano AZURE_OPENAI_ENDPOINT o AZURE_OPENAI_API_KEY nel .env", file=sys.stderr)
        sys.exit(1)

    pdf_files = [KNOWLEDGE_BASE_DIR / name for name in TARGET_FILES]
    missing = [f.name for f in pdf_files if not f.exists()]
    if missing:
        print(f"❌ File non trovati in {KNOWLEDGE_BASE_DIR}:", file=sys.stderr)
        for m in missing:
            print(f"   • {m}", file=sys.stderr)
        sys.exit(1)

    print(f"📂 File da caricare:")
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
    AZURE_PREVENTIVI_VECTOR_STORE_ID={vs.id}
""")


if __name__ == "__main__":
    main()
