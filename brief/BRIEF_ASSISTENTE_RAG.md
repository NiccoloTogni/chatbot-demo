# Brief: Aggiunta modalità "Assistente RAG" alla demo

## 1. Contesto

Nella repo esiste già una demo funzionante di chatbot **"vanilla"** che usa Azure OpenAI (deployment `gpt-5.4`). Stack attuale:

- **Backend**: Python 3.12 + FastAPI
- **Frontend**: `index.html` con JS embedded (no framework)
- **Deploy**: Azure Web App `vanilla-chatbot`, GitHub Actions (`deploy.yml` zip-and-push, niente build steps custom)
- **Secrets**: env vars Azure OpenAI gestite come **App Settings di Azure**, lette a runtime via `os.environ`. Il workflow contiene solo `AZUREAPPSERVICE_PUBLISHPROFILE` per il deploy.

**Obiettivo**: aggiungere una **seconda modalità "Assistente RAG"** affiancata alla vanilla. Stesso modello (`gpt-5.4`), stessa UI di chat, ma con accesso a una knowledge base di documenti aziendali. Le risposte devono includere **citazioni alle fonti** in modo visibile.

La knowledge base sono 6 PDF già pronti che vanno messi in `data/knowledge_base/`:

```
PQ-04_NonConformita_Fornitori.pdf
ST-MAT-01_Materiali_Approvati.pdf
VRB-2025-03_Riunione_Qualita.pdf
IO-PROD-07_Stampaggio_Iniezione.pdf
HR-POL-03_Rimborsi_Trasferte.pdf
HR-FAQ-01_Onboarding.pdf
```

## 2. Decisioni architetturali (già prese — non rimettere in discussione)

- **API**: Azure OpenAI **Responses API** (è il successore della Assistants API, deprecata)
- **Tool**: `file_search` — gestisce nativamente upload, chunking, embeddings, retrieval e citations. **Niente vector DB esterno, niente librerie di embeddings, niente chunking custom.**
- **Vector store**: **uno solo** con tutti i 6 documenti dentro (vedi gotcha #1)
- **Modello**: stesso deployment `gpt-5.4` del vanilla (non creare nuove deployment)

## 3. Gotchas critici (leggere PRIMA di scrivere codice)

1. **Single vector store su Azure**: il parametro `vector_store_ids` di `file_search` accetta un array, ma Azure OpenAI attualmente onora **un solo vector store per chiamata** (limitazione documentata nei forum Microsoft Q&A — comportamento diverso da OpenAI diretto). Non tentare di dividere i documenti in più store.

2. **API version**: serve un'`api-version` preview che supporti Responses API + vector stores. Default suggerito: `2025-04-01-preview`. **Verificare la versione più recente disponibile** controllando https://learn.microsoft.com/en-us/azure/ai-services/openai/reference-preview prima del commit. Se il vanilla usa già un'`api-version`, valutare se è compatibile con Responses API o se va bumpata (eventualmente env var separata `AZURE_OPENAI_API_VERSION_RESPONSES` per non rompere il vanilla).

3. **Citations format**: il `file_search` restituisce le citazioni come `annotations` dentro l'output text. Vanno estratte iterando sulla struttura della response (consultare docs SDK Python più recenti — la struttura tipica è `response.output[i].content[j].annotations[k]` con campi `file_id`, `index`, ecc.). Mappare `file_id` → nome file leggibile per la UI.

4. **SDK Python**: usare il pacchetto `openai` (non esiste un `azure-openai` separato). Client da istanziare come `AzureOpenAI(...)`. **Replica esattamente lo stesso pattern di inizializzazione che già usa il chatbot vanilla nella repo** — non reinventare. Verifica solo che la versione di `openai` in `requirements.txt` supporti `client.responses.create()` e `client.vector_stores.*` (>= 1.66 circa, ma verifica al momento dell'implementazione).

## 4. Setup iniziale

### 4.1 Dipendenze (`requirements.txt`)
- Verificare che `openai` sia presente e a versione recente. Se necessario bumpare. Non aggiungere altre dipendenze (no chromadb, no langchain, no sentence-transformers — tutto è gestito da Azure).

### 4.2 Env vars da aggiungere
Una sola nuova App Setting su Azure (e su `.env` locale per sviluppo):

- `AZURE_VECTOR_STORE_ID` — popolata dopo aver eseguito lo script di setup (Task 1)

Eventualmente, se l'API version del vanilla non è compatibile con Responses API:
- `AZURE_OPENAI_API_VERSION_RESPONSES` — es. `"2025-04-01-preview"`

L'utente aggiungerà manualmente queste App Settings nel portale Azure dopo il setup. **Non serve modificare `deploy.yml`.**

## 5. Task da eseguire (in ordine)

### Task 1 — Script di setup vector store (one-off, gira in locale)

Creare `scripts/setup_vector_store.py` che:

1. Si connette ad Azure OpenAI usando le stesse env vars del vanilla
2. Crea un vector store chiamato `baldan-knowledge-base`
3. Carica tutti i PDF dalla directory `data/knowledge_base/` usando il batch helper (`vector_stores.file_batches.upload_and_poll`)
4. Stampa il `vector_store_id` con un messaggio chiaro tipo:
   ```
   ✅ Vector store creato: vs_abc123xyz
   Aggiungi al .env e alle App Settings di Azure:
       AZURE_VECTOR_STORE_ID=vs_abc123xyz
   ```
5. Stampa anche lo stato di processing dei file (es. "6/6 file completed")

Lo script deve essere **idempotente-friendly**: se eseguito di nuovo, crea un nuovo vector store (non aggiorna quello vecchio). Non implementare logica di delete/replace — l'utente gestirà manualmente i vecchi store nel portale se necessario. Aggiungere un commento in cima allo script che lo specifichi.

### Task 2 — Backend handler RAG

Localizzare l'endpoint chat esistente del vanilla e:

**Opzione A (preferita)**: estendere l'endpoint esistente con un parametro `mode: "vanilla" | "rag"`. Se `mode="rag"`, usa il flusso Responses API + file_search. Altrimenti il flusso vanilla esistente.

**Opzione B**: se l'endpoint esistente è troppo specifico per la modalità vanilla, creare un nuovo endpoint `/api/chat/rag`. Ma preferire A se possibile per minimizzare la duplicazione frontend.

L'handler RAG deve:

1. Ricevere la domanda dell'utente (e opzionalmente la conversation history, se il vanilla la gestisce)
2. Chiamare `client.responses.create()` con:
   - `model`: stesso deployment del vanilla
   - `input`: la domanda (o la history mappata nel formato Responses API)
   - `tools`: `[{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}]`
   - `include`: includere i risultati di search per poter mostrare le citation con preview (opzionale ma consigliato)
3. Aggiungere un **system prompt** specifico per la modalità RAG, qualcosa tipo:
   > "Sei un assistente aziendale. Rispondi alle domande basandoti ESCLUSIVAMENTE sui documenti forniti tramite il tool file_search. Se l'informazione non è nei documenti, dillo esplicitamente. Cita sempre le fonti. Rispondi in italiano."
4. Estrarre dalla response:
   - Il testo della risposta (`output_text` o equivalente)
   - Le **annotations/citations**: per ognuna ricavare il `file_id` e mapparlo al nome file leggibile (es. "PQ-04 Non Conformità Fornitori"). Se possibile estrarre anche lo snippet di testo che ha generato la citation.
5. Restituire JSON:
   ```json
   {
     "answer": "Secondo la procedura PQ-04...",
     "citations": [
       {
         "file_name": "PQ-04 Non Conformità Fornitori",
         "snippet": "Il Responsabile Qualità classifica la NC in tre livelli..."
       }
     ]
   }
   ```

**Mapping `file_id` → nome leggibile**: dato che i file sono caricati da noi in setup, possiamo o (a) salvare un dict `file_id → display_name` in fase di setup e leggerlo dall'env/config, oppure (b) recuperare on-the-fly i metadati del file con `client.files.retrieve(file_id)` (più lento ma più semplice). Preferire (b) per la prima implementazione, poi cachare in memoria con `@lru_cache`.

### Task 3 — Frontend: aggiungere tab RAG

In `index.html`:

1. Aggiungere sopra l'area chat **due tab**: "Vanilla" (selezionata di default) e "Assistente RAG"
2. Lo styling delle tab deve essere semplice e coerente con quello esistente (no librerie nuove)
3. Mantenere **un'unica area di chat** sotto le tab — non duplicare la UI. La tab attiva determina solo a quale endpoint/mode viene mandato il messaggio successivo.
4. Quando l'utente cambia tab, **pulire la conversazione corrente** (oppure mantenerle separate in due array — scelta di Claude Code in base a quanto è semplice). Nel dubbio: pulire, è più semplice e non confonde l'utente in demo.
5. Mostrare un piccolo **badge/indicator** che segnala la modalità attiva (es. icona o label "🔍 Modalità RAG attiva — risposte basate sui documenti aziendali")

### Task 4 — Frontend: render delle citations

Sotto ogni risposta in modalità RAG, mostrare le citazioni in modo visibile:

```
[risposta del modello]

📚 Fonti:
  • PQ-04 Non Conformità Fornitori
  • VRB 2025-03 Riunione Qualità
```

Se ci sono snippet disponibili, le citazioni devono essere **espandibili** (click per aprire un piccolo box con il testo del chunk citato). Stile semplice, niente librerie esterne. CSS inline o `<style>` nel `<head>` come probabilmente fa già il resto della pagina.

**Importante per l'effetto demo**: le citations devono essere immediatamente visibili e ben riconoscibili — sono il punto narrativo principale della demo (la differenza tra "AI generica" e "AI che conosce la tua azienda"). Non nasconderle in un menu o in fondo.

### Task 5 — Deploy

1. Aggiornare `requirements.txt` se è stata bumpata la versione di `openai`
2. Committare tutto inclusi i 6 PDF in `data/knowledge_base/` (~150KB totali, accettabili nello zip di deploy)
3. Push su `master` → il workflow esistente fa il resto
4. **Manualmente**: aggiungere `AZURE_VECTOR_STORE_ID` (e eventuale `AZURE_OPENAI_API_VERSION_RESPONSES`) come App Settings nel portale Azure della Web App `vanilla-chatbot`. Riavviare l'app.

Documentare questi step manuali in un breve `README_RAG.md` o in una sezione del README esistente, includendo l'ordine: (1) eseguire `python scripts/setup_vector_store.py` localmente, (2) copiare il `vector_store_id` ottenuto, (3) aggiungerlo come App Setting su Azure.

## 6. Verifica funzionale

Dopo il deploy, testare in entrambe le modalità queste 5 domande (sono quelle dell'esercizio in aula):

1. *"Qual è la procedura per gestire una non conformità fornitore?"*
   → Vanilla: risposta generica. RAG: deve citare **PQ-04**.

2. *"Quali materiali sono approvati per componenti a contatto con alimenti?"*
   → RAG: deve citare **ST-MAT-01** (sezione 4) e menzionare PP-003, TPE-003.

3. *"Cosa è stato deciso nell'ultima riunione qualità riguardo ai fornitori in ritardo?"*
   → RAG: deve citare **VRB-2025-03** e menzionare il piano miglioramento Fornitore A.

4. *"Quali sono i requisiti di umidità residua per il PP compound in accettazione?"*
   → RAG: deve citare **PQ-04** (max 0.10% per PP, 0.05% per PP-GF30) e idealmente anche **ST-MAT-01** o **IO-PROD-07** che ne parlano.

5. *"Quanto posso spendere per una cena in trasferta a Milano?"*
   → RAG: deve citare **HR-POL-03** (€30 per la cena, €130/notte per Milano).

In tutte le risposte RAG, le **citation devono essere visibili e cliccabili** se gli snippet sono disponibili.

## 7. Riferimenti

- **Azure OpenAI Responses API**: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference-preview
- **File search tool (OpenAI docs, applica anche ad Azure)**: https://platform.openai.com/docs/guides/tools-file-search
- **Vector stores Azure**: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/file-search
- **Limitazione single vector store su Azure**: https://learn.microsoft.com/en-us/answers/questions/5540289/responses-api-searches-only-one-attached-vector-st

## 8. Out of scope (NON fare)

- Streaming responses (lasciar perdere per la prima versione, può essere aggiunto dopo)
- Conversation memory persistente lato server (mantenere lo stesso modello del vanilla)
- Autenticazione utenti
- Rate limiting / cost monitoring
- Vector DB self-hosted o alternative (no Qdrant, ChromaDB, Pinecone)
- Reindex automatico dei documenti (lo script di setup è manuale, va benissimo per la demo)
- Multi-language detection (i documenti sono in italiano, le domande in italiano)