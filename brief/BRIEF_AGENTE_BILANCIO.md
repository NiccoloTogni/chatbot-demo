# Brief: Aggiunta tab "Agente Bilancio" alla demo

## 1. Contesto

Nella web app esiste già:
- **Tab "Vanilla"**: chatbot generico Azure OpenAI
- **Tab "Assistente RAG"**: chatbot con file_search su knowledge base aziendale

Stack: **Python 3.12 + FastAPI** backend, **index.html con JS embedded** frontend, deploy su Azure Web App `vanilla-chatbot` via GitHub Actions zip-and-push.

**Obiettivo**: aggiungere una **terza tab "Agente Bilancio"** che dimostra il livello 4 della scala (slide 9 della presentazione). Mostra un agente che esegue un workflow multi-step su un PDF di bilancio: estrae dati, calcola KPI, genera grafici, scrive commento, produce report PDF scaricabile.

## 2. Decisioni architetturali (già prese — non rimettere in discussione)

- **NO agent loop autonomo**, NO function calling con LLM che decide cosa fare. Il workflow è **orchestrato dal backend** in modo deterministico: 4 step fissi, in ordine fisso. L'AI viene chiamata solo per i task in cui serve (estrazione + commento), non per "decidere il prossimo passo".
- **NO streaming / SSE**. Il frontend chiama 4 endpoint in sequenza, mostrando il progresso step-by-step nella UI man mano che riceve le risposte. Più semplice, più robusto, più controllabile in demo dal vivo.
- **Stessa Responses API** già usata per la modalità RAG (Azure OpenAI, deployment `gpt-5.4`).
- **Estrazione PDF**: Responses API riceve direttamente il PDF come input (no `pdfplumber`, no parsing custom). gpt-5.4 lo legge nativamente.
- **Output PDF**: generato con `reportlab` (già usato in altre parti del progetto, dipendenza standard).
- **Sessioni**: in-memory dict nel backend, identificate da `session_id`. Niente persistenza, niente database. Cleanup automatico dopo 30 minuti.

## 3. Workflow (4 step)

| Step | Tipo | Cosa fa | Tempo atteso |
|------|------|---------|--------------|
| 1 | AI | Estrae dati dal PDF in JSON strutturato | 5-15 sec |
| 2 | Deterministico | Calcola KPI e variazioni | <1 sec |
| 3 | Deterministico | Genera 3 grafici PNG con matplotlib | 1-2 sec |
| 4 | AI + det. | Scrive commento AI + compone report PDF | 10-20 sec |

**Totale percepito in aula**: 20-40 secondi. Sufficiente per commentare ogni step a voce mentre esegue.

## 4. Task da eseguire (in ordine)

### Task 1 — Backend: setup e nuovo modulo

Creare un nuovo modulo `agent_bilancio.py` (o nome equivalente, in linea con la convenzione esistente). Il modulo contiene la logica dei 4 step. Gli endpoint FastAPI possono stare nel main router o in un router separato `/api/agent/*`.

**Sessione in memoria**: dict globale `SESSIONS: dict[str, dict]` dove ogni entry contiene:
```python
{
    "created_at": datetime,
    "pdf_file_id": str,        # file_id Azure dopo upload
    "extracted_data": dict,    # output step 1
    "kpis": dict,              # output step 2
    "chart_paths": list[str],  # output step 3
    "report_path": str,        # output step 4
}
```

Cleanup: una funzione `cleanup_old_sessions()` chiamata all'inizio di ogni endpoint che rimuove sessioni più vecchie di 30 min e i file associati.

**File temporanei**: salvare in `/tmp/agent_sessions/{session_id}/` (su Azure Web App `/tmp` esiste ed è scrivibile). Non in `/home` per evitare di consumare lo storage persistente.

### Task 2 — Endpoint upload (Step 0)

`POST /api/agent/upload`

- Riceve un PDF via multipart form-data
- Valida che sia un PDF (primi byte `%PDF`)
- Limite dimensione: 10 MB (più che sufficiente per un bilancio)
- Carica il file su Azure tramite `client.files.create(file=..., purpose="assistants")` per ottenere un `file_id` riusabile
- Crea una nuova entry in `SESSIONS` con `session_id` UUID
- Restituisce: `{ "session_id": "...", "filename": "..." }`

### Task 3 — Step 1: estrazione dati (AI)

`POST /api/agent/step1?session_id=...`

Chiamata `client.responses.create()` con:
- `model`: stesso deployment del Vanilla/RAG
- `input`: lista di messaggi con un `input_file` che riferisce il `pdf_file_id` della sessione + un prompt di estrazione (vedi sotto)
- `response_format`: `{ "type": "json_object" }` se supportato dalla versione SDK, altrimenti istruire l'AI nel prompt a rispondere solo JSON puro

**Prompt esatto da usare** (system + user):

```
SYSTEM:
Sei un analista finanziario esperto. Estrai i dati di bilancio dal PDF fornito
e restituisci ESCLUSIVAMENTE un oggetto JSON valido (no markdown, no commenti).

Lo schema richiesto è:
{
  "azienda": "<nome società>",
  "anni": [<lista anni disponibili, es. [2023, 2024, 2025]>],
  "conto_economico": {
    "<anno>": {
      "ricavi": <numero in migliaia di euro>,
      "costi_materie_prime": <numero negativo>,
      "costi_energia": <numero negativo>,
      "costi_personale": <numero negativo>,
      "altri_costi_operativi": <numero negativo>,
      "ebitda": <numero>,
      "ammortamenti": <numero negativo>,
      "ebit": <numero>,
      "oneri_finanziari": <numero negativo>,
      "imposte": <numero negativo>,
      "utile_netto": <numero>
    }
  },
  "stato_patrimoniale": {
    "<anno>": {
      "immobilizzazioni_nette": <numero>,
      "capitale_circolante_netto": <numero>,
      "capitale_investito_netto": <numero>,
      "patrimonio_netto": <numero>,
      "posizione_finanziaria_netta": <numero>
    }
  },
  "note_qualitative": "<sintesi in 2-3 frasi delle note esplicative se presenti, altrimenti stringa vuota>"
}

Tutti i numeri devono essere in MIGLIAIA DI EURO come interi (non stringhe).
I costi vanno indicati con segno NEGATIVO.
Se un dato non è disponibile nel PDF, usa null (non zero).

USER:
Estrai i dati di bilancio dal documento allegato.
```

Salvare il JSON estratto in `SESSIONS[session_id]["extracted_data"]`.

Restituire al frontend una versione **sintetica** per la UI:
```json
{
  "step": 1,
  "status": "completed",
  "summary": "Estratti dati per gli anni 2023, 2024, 2025 — PolimerTech Italia S.p.A.",
  "details": ["Conto Economico: 11 voci × 3 anni", "Stato Patrimoniale: 5 voci × 3 anni"]
}
```

**Gestione errori**: se l'AI ritorna JSON malformato, ritentare 1 volta con un prompt rinforzato. Se fallisce ancora, restituire 500 con messaggio chiaro al frontend (che mostrerà errore visibile invece di stallo).

### Task 4 — Step 2: calcolo KPI (deterministico)

`POST /api/agent/step2?session_id=...`

Codice Python puro che legge `extracted_data` e calcola:

**Per il conto economico**:
- EBITDA margin (EBITDA / Ricavi) per anno
- EBIT margin per anno
- Net margin per anno
- Variazioni YoY (%) di Ricavi, EBITDA, EBIT, Utile netto
- CAGR ricavi sul periodo

**Per lo stato patrimoniale e indicatori**:
- PFN / EBITDA per anno
- PFN / Patrimonio Netto per anno
- ROI (EBIT / CIN) per anno
- ROE (Utile / PN) per anno
- Copertura interessi (EBITDA / |Oneri finanziari|) per anno

Salvare tutto in `SESSIONS[session_id]["kpis"]` come dict strutturato.

**Gestione divisioni per zero**: usare `None` come valore se denominatore è 0 o None. Mai eccezioni.

Restituire summary al frontend tipo:
```json
{
  "step": 2,
  "status": "completed",
  "summary": "Calcolati 12 KPI principali",
  "details": [
    "EBITDA margin 2025: 9.0%",
    "PFN/EBITDA 2025: 4.2x",
    "Variazione ricavi 25/24: +17.0%"
  ]
}
```

### Task 5 — Step 3: generazione grafici (deterministico)

`POST /api/agent/step3?session_id=...`

Tre grafici matplotlib salvati come PNG (300 DPI) in `/tmp/agent_sessions/{session_id}/`:

1. **`trend_ricavi_ebitda.png`** — bar chart raggruppato: ricavi e EBITDA per i 3 anni, con asse Y in M€. Mostra visivamente la divergenza (ricavi su, EBITDA piatto/giù).

2. **`marginalita.png`** — line chart con 3 linee: EBITDA margin %, EBIT margin %, Net margin %, sui 3 anni. Mostra il declino della marginalità.

3. **`struttura_finanziaria.png`** — combo: bar chart con PFN e PN per anno (asse sinistro, M€) + line per il rapporto PFN/EBITDA (asse destro, multiplo). Mostra l'aumento della leva.

**Stile grafici**: usa una palette professionale e sobria (es. `#1a3a52`, `#2d6a8f`, `#a02020` per attenzione). Font Helvetica/Arial, dimensione 14×8 cm, titoli in italiano. Aggiungi gridlines orizzontali leggere. NIENTE emoji nei grafici.

Salvare i path in `SESSIONS[session_id]["chart_paths"]`.

Restituire al frontend:
```json
{
  "step": 3,
  "status": "completed",
  "summary": "Generati 3 grafici",
  "details": ["Trend ricavi vs EBITDA", "Marginalità", "Struttura finanziaria"]
}
```

### Task 6 — Step 4: commento AI + composizione PDF

`POST /api/agent/step4?session_id=...`

**Sotto-step 4a — Commento AI**:

Chiamata a Responses API con un prompt che include i KPI calcolati e le note qualitative estratte. Prompt:

```
SYSTEM:
Sei un analista finanziario senior. Devi scrivere un commento professionale
e sintetico su un bilancio aziendale, in italiano, basandoti sui dati forniti.

Il commento deve essere strutturato in queste sezioni esatte:

1. EXECUTIVE SUMMARY (3-4 frasi, sintesi della situazione)
2. PUNTI DI FORZA (2-3 bullet, ciò che va bene)
3. AREE DI ATTENZIONE (3-4 bullet, criticità con riferimento a dati specifici)
4. RACCOMANDAZIONI (3-4 bullet, azioni concrete e prioritarie)

Usa un tono professionale ma chiaro. Cita SEMPRE i numeri specifici per
sostanziare ogni punto (es. "EBITDA margin sceso dal 15.0% al 9.0% in due anni").
Non inventare dati non presenti nel materiale fornito.

Restituisci ESCLUSIVAMENTE un oggetto JSON con questa struttura:
{
  "executive_summary": "<paragrafo>",
  "punti_di_forza": ["<bullet>", ...],
  "aree_di_attenzione": ["<bullet>", ...],
  "raccomandazioni": ["<bullet>", ...]
}

USER:
Ecco i dati di bilancio:
KPI: <serializza qui SESSIONS[session_id]["kpis"] in JSON>
Note qualitative dal bilancio: <SESSIONS[session_id]["extracted_data"]["note_qualitative"]>

Scrivi il commento.
```

**Sotto-step 4b — Composizione PDF con reportlab**:

Layout del report PDF (4-5 pagine):

- **Copertina**: titolo "Report di Analisi Finanziaria" + nome azienda + periodo + data generazione + footer "Documento generato automaticamente da agente AI — verifica sempre i dati con un professionista"
- **Pagina 1 — Executive Summary**: testo del summary in box evidenziato
- **Pagina 2 — KPI principali**: tabella con tutti i KPI calcolati (3 anni in colonne)
- **Pagina 3 — Grafici**: i 3 grafici PNG inseriti, ciascuno con didascalia
- **Pagina 4 — Analisi**: punti di forza + aree di attenzione + raccomandazioni in liste puntate

Stile coerente con il PDF di input (font Helvetica, palette `#1a3a52` per header, tabelle con header colorato e righe alternate).

Salvare in `/tmp/agent_sessions/{session_id}/report.pdf` e in `SESSIONS[session_id]["report_path"]`.

Restituire al frontend:
```json
{
  "step": 4,
  "status": "completed",
  "summary": "Report pronto",
  "details": ["4 pagine", "3 grafici", "12 KPI"],
  "download_url": "/api/agent/download/{session_id}"
}
```

### Task 7 — Endpoint download

`GET /api/agent/download/{session_id}`

- Verifica che la sessione esista e abbia `report_path`
- Restituisce il PDF con `Content-Type: application/pdf` e `Content-Disposition: attachment; filename="report_polimertech.pdf"` (o nome dinamico basato sul nome azienda estratto)

### Task 8 — Frontend: terza tab

In `index.html`:

1. Aggiungere terza tab "🤖 Agente Bilancio" accanto a Vanilla e RAG, con stile coerente
2. Quando attivata, **NON mostrare la chat**. Mostrare invece un pannello dedicato con:
   - Box upload file (drag&drop o button) con label "Carica un bilancio in PDF"
   - Dopo upload: pulsante "▶ Avvia analisi"
   - Lista dei 4 step come progress checklist (vedi Task 9)
   - Quando completati tutti gli step: pulsante "📥 Scarica report"
3. Reset / nuova analisi: pulsante per ricominciare con un altro file

Niente librerie esterne nuove. Solo HTML + CSS + JS vanilla, coerente con il resto della pagina.

### Task 9 — Frontend: progress checklist

Mostrare i 4 step come checklist visiva, ognuno con tre stati:
- ⏸ in attesa (grigio)
- ⏳ in corso (animazione spinner)
- ✓ completato (verde) + summary string ricevuta dal backend
- ✗ errore (rosso) + messaggio errore

Layout suggerito:

```
[ ✓ ] Step 1 · Estrazione dati
       Estratti dati per gli anni 2023, 2024, 2025 — PolimerTech Italia S.p.A.

[ ✓ ] Step 2 · Calcolo KPI
       Calcolati 12 KPI principali

[⏳] Step 3 · Generazione grafici
       In corso...

[ ⏸ ] Step 4 · Analisi e composizione report
```

Il flusso JS è sequenziale: chiama step1, alla risposta aggiorna UI, chiama step2, ecc. Se uno step fallisce (response 4xx/5xx), ferma il workflow e mostra errore visibile.

### Task 10 — Deploy

1. Aggiornare `requirements.txt` se servono dipendenze mancanti (probabile: `matplotlib` per i grafici se non già presente; `reportlab` dovrebbe esserci già)
2. **Nessuna nuova env var**: usa quelle esistenti di Azure OpenAI
3. **Nessuna modifica al `deploy.yml`**
4. Push su `master` → workflow esistente fa il resto

## 5. Verifica funzionale

Test in aula con il file di esempio `Bilancio_PolimerTech_2023-2025.pdf`:

1. Carica il PDF → step 0 OK
2. Click "Avvia analisi" → i 4 step si succedono, ognuno con summary visibile
3. Tempo totale: 20-40 secondi
4. Download del report PDF
5. Aprendo il report:
   - Executive summary deve menzionare la **divergenza ricavi/marginalità** (ricavi +34% ma EBITDA -19%)
   - Aree di attenzione devono includere **PFN/EBITDA salito a 4.2x** sopra soglia covenant
   - Raccomandazioni devono includere qualcosa su **pricing power** o **hedging energia** (l'AI dovrebbe inferirlo dalle note qualitative)
   - I 3 grafici devono essere presenti e leggibili

Edge case da testare:
- Upload di un file non-PDF → errore visibile, no crash
- Upload di un PDF non-bilancio (es. una fattura) → step 1 dovrebbe fallire o restituire JSON con dati null → UI mostra errore comprensibile
- Refresh della pagina a metà workflow → sessione persa, si ricomincia (accettabile per demo)

## 6. Out of scope (NON fare)

- Streaming / SSE (esplicitamente escluso)
- Function calling autonomo / agent loop
- Persistenza sessioni su DB
- Login / multi-tenant
- Cache risultati per file uguali
- Supporto altre lingue oltre italiano
- Confronto multi-bilancio (un solo PDF alla volta)
- Editing del prompt da UI
- Streaming dei grafici (statici basta)

## 7. Riferimenti

- **Responses API con file input**: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses
- **File upload via Files API Azure**: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#files
- **reportlab Platypus** (per generazione PDF strutturati): https://docs.reportlab.com/reportlab/userguide/ch5_platypus/
- **matplotlib styling**: https://matplotlib.org/stable/users/explain/customizing.html
