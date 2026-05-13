# Vanilla LLM Demo

Webapp didattica per il corso AI. Espone una chat verso un LLM "puro" (Azure
OpenAI) **senza nessun tool**: niente ricerca web, niente calcolatrice, niente
accesso a documenti. Serve a mostrare in aula, dal vivo, i comportamenti di
un modello quando non è aiutato da niente — allucinazioni, errori sui
conteggi, mancanza di informazioni recenti, errori di calcolo.

Stato della conversazione **lato browser in `sessionStorage`**: sopravvive al
refresh della pagina, viene cancellato alla chiusura della tab. Il backend è
stateless — riceve l'intera cronologia ad ogni chiamata.

## Architettura

```
.
├── app.py                 # entry point per App Service (importa backend.main)
├── requirements.txt       # dipendenze (richiesto in root da App Service)
├── backend/
│   ├── main.py            # ~200 righe FastAPI
│   ├── requirements.txt   # copia per dev locale
│   └── .env.example       # variabili da copiare in .env
├── frontend/
│   └── index.html         # SPA single-file (HTML+CSS+JS)
└── README.md
```

Una sola applicazione Python serve sia le API (`/api/*`) sia il frontend
statico (`/`). Un solo dominio, un solo deploy.

## Quale modello Azure OpenAI usare?

Per la **didattica**, il modello migliore è uno **non-reasoning di taglia
medio-piccola**, perché amplifica i comportamenti vanilla che vogliamo
mostrare:

- ✅ Consigliato: `gpt-4o-mini`, `gpt-35-turbo`
- ⚠️ Sconsigliato: `gpt-5`, `o1`, `o3` — i modelli con reasoning interno
  "smussano" molti errori e rendono la demo meno efficace

## Prerequisiti

- Una sottoscrizione Azure con accesso al portale
- Una risorsa **Azure OpenAI** già creata, con almeno un deployment di un
  modello chat (es. `gpt-4o-mini`)
- Le credenziali della risorsa Azure OpenAI a portata di mano:
  - **Endpoint**: tipo `https://<nome-risorsa>.openai.azure.com/`
  - **API Key**: una delle due chiavi sotto "Keys and Endpoint"
  - **Deployment name**: il nome che hai dato al deployment del modello
    (NON il nome del modello base)

---

# Deploy su Azure App Service — guida passo passo

## Passo 1 · Carica il codice su un repo Git (consigliato)

App Service può fare deploy direttamente da:
- **GitHub** (più comodo, deploy automatico ad ogni push)
- **Azure DevOps Repos**
- **Zip upload manuale** (vedi alternativa più sotto)

Crea un repo (anche privato) con il contenuto di questa cartella. Se preferisci
saltare questo passo, vedi *Alternativa: deploy da zip* alla fine.

## Passo 2 · Crea l'App Service dal portale

Dal portale Azure:

1. **Crea una risorsa → Web App**
2. Compila i campi:
   - **Subscription**: la tua
   - **Resource Group**: usa lo stesso della risorsa Azure OpenAI (consigliato),
     oppure creane uno nuovo (es. `rg-corso-ai-rossi`)
   - **Name**: nome univoco globalmente, sarà parte dell'URL
     (es. `rossi-vanilla-llm` → `https://rossi-vanilla-llm.azurewebsites.net`)
   - **Publish**: `Code` (non Container)
   - **Runtime stack**: `Python 3.12`
   - **Operating System**: `Linux`
   - **Region**: una vicina (es. `Italy North` o `West Europe`)
3. **Pricing plan**: per uso didattico saltuario va benissimo **B1 Basic**
   (~13 €/mese, fermabile quando non serve). Evita il piano Free F1: ha
   limiti di CPU che possono dare timeout sulle risposte degli LLM.
4. Lascia il resto ai default e clicca **Review + Create → Create**.

Aspetta 1-2 minuti che il deploy della risorsa finisca.

## Passo 3 · Configura le variabili d'ambiente

Una volta creata l'App Service:

1. Vai sulla risorsa appena creata
2. Menu laterale: **Settings → Environment variables**
3. Sezione **App settings**, clicca **+ Add** per ognuna di queste:

| Name | Value |
|------|-------|
| `AZURE_OPENAI_ENDPOINT` | `https://<nome-risorsa>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | la tua API key Azure OpenAI |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | nome del deployment (es. `gpt-4o-mini`) |
| `ACCESS_TOKEN` | password per gli studenti (es. `corso-rossi-2026`) |
| `MAX_OUTPUT_TOKENS` | `800` |
| `MAX_HISTORY_MESSAGES` | `30` |
| `RATE_LIMIT` | `30/minute` |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` |
| `WEBSITES_PORT` | `8000` |

L'ultima riga (`SCM_DO_BUILD_DURING_DEPLOYMENT=true`) dice ad App Service di
installare automaticamente le dipendenze dal `requirements.txt` durante il
deploy. È **fondamentale**.

`WEBSITES_PORT=8000` indica la porta su cui ascolta l'app — corrisponde a
quella usata da uvicorn nel comando di startup del passo successivo.

Salva con **Apply** in fondo alla pagina. App Service riavvia.

## Passo 4 · Imposta lo Startup Command

App Service per Python ha un default che funziona solo per Flask/Django. Per
FastAPI dobbiamo dire esplicitamente come avviare l'app.

1. Sempre nella risorsa App Service, menu laterale: **Settings → Configuration**
2. Tab **General settings**
3. Campo **Startup Command**, incolla:

   ```
   gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120
   ```

   Spiegazione:
   - `app:app` → modulo `app.py`, oggetto `app` (il nostro wrapper)
   - `-k uvicorn.workers.UvicornWorker` → usa il worker async di uvicorn
     (FastAPI è async)
   - `--timeout 120` → permette risposte lunghe dal modello (2 minuti max)

4. **Save** in alto. App Service riavvia di nuovo.

## Passo 5 · Collega il repo per il deploy

Sempre nella risorsa App Service:

1. Menu laterale: **Deployment → Deployment Center**
2. **Source**: scegli `GitHub` (o `Azure Repos`)
3. Autentica con il tuo account
4. Seleziona organizzazione, repo e branch (di solito `main`)
5. **Build provider**: `App Service build service` (più semplice)
6. **Save**

Il primo deploy parte automaticamente. Lo trovi nel tab **Logs** della stessa
schermata. Aspetta che lo stato diventi `Success` (~3-5 minuti la prima volta).

## Passo 6 · Verifica che funzioni

Apri `https://<nome-app>.azurewebsites.net/api/health` — dovresti vedere:

```json
{"status": "ok"}
```

Poi apri `https://<nome-app>.azurewebsites.net/` — dovresti vedere la
schermata password.

Inserisci `ACCESS_TOKEN` e prova a mandare il messaggio:
*"Quante R ci sono in STRAWBERRY?"*

Se torna una risposta (probabilmente sbagliata, è il punto), tutto funziona.

## Se qualcosa non va

**Errore 500 / pagina di default Azure**

Vai su **Monitoring → Log stream** della risorsa App Service e leggi gli
errori in tempo reale. Le cause più comuni:
- Variabili d'ambiente mancanti o errate (controlla soprattutto endpoint e key)
- `SCM_DO_BUILD_DURING_DEPLOYMENT` non settato a `true` → mancano le
  dipendenze; aggiungi la variabile e fai un deploy nuovo (Deployment Center →
  Sync)
- Startup command non impostato correttamente

**Errore 401 dalla chat ma con password giusta**

Hai cambiato `ACCESS_TOKEN` dopo il primo accesso. Apri la console del browser
(F12), vai su **Application → Session Storage**, cancella tutte le voci
`vanilla-llm-demo:*`, ricarica e reinserisci la nuova password.

**Errore 502 o timeout sulla chat**

L'`API_VERSION` Azure OpenAI potrebbe non supportare il modello che hai
deployato. Prova ad aggiornare `AZURE_OPENAI_API_VERSION` all'ultima
disponibile per la tua risorsa (la trovi nel portale Azure OpenAI Studio).

## Alternativa: deploy da zip (senza Git)

Se non vuoi usare un repo:

1. Comprimi il contenuto della cartella in un file `.zip` (NON la cartella —
   il contenuto: `app.py`, `requirements.txt`, `backend/`, `frontend/`,
   `README.md` devono essere in root dello zip)
2. Apri Azure Cloud Shell dal portale (icona `>_` in alto a destra)
3. Carica lo zip cliccando l'icona "Manage files → Upload" nella Cloud Shell
4. Esegui (sostituisci nome-rg, nome-app, nome-zip):

   ```bash
   az webapp deploy \
     --resource-group <nome-rg> \
     --name <nome-app> \
     --src-path <nome-zip>.zip \
     --type zip
   ```

Le variabili d'ambiente e lo startup command li imposti come nei Passi 3 e 4.

---

# Sviluppo locale (opzionale)

Per testare prima del deploy:

```bash
cd vanilla-llm-demo
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Compila il file .env (vedi backend/.env.example)
cp backend/.env.example backend/.env
# poi compila i valori

# Avvia
uvicorn app:app --reload --port 8000
```

Apri `http://localhost:8000`. Lasciando `ACCESS_TOKEN` vuoto in `.env`, l'auth
è disattivata.

---

# Sicurezza

- L'API key di Azure OpenAI **non lascia mai il server**. Il browser non la
  vede mai.
- L'auth è una password condivisa (`ACCESS_TOKEN`) inviata come Bearer token.
  È **sufficiente per uso didattico in aula**, **non sufficiente** per
  esposizione pubblica permanente.
- Per uso prolungato, attivare l'autenticazione integrata di App Service:
  **Settings → Authentication → Add identity provider** (Microsoft, Google,
  ecc.). Disattiva la nostra auth a token settando `ACCESS_TOKEN=` (vuoto), e
  lascia gestire tutto a Azure.
- Rate limit attivo per IP (default 30/min) per evitare consumo eccessivo di
  token.

# Costi attesi durante un corso

Stima per **7 partecipanti × 30 minuti di esercizio attivo × `gpt-4o-mini`**:
- ~250 chiamate totali
- ~100k token totali (input + output)
- **Costo Azure OpenAI: < 0.50 €**
- **Costo App Service B1**: ~0.40 € se acceso solo durante le 6h del corso,
  ~13 € se lasciato sempre acceso un mese

Suggerimento: stoppa la risorsa App Service quando non serve (**Overview →
Stop**). I costi sono in pausa, basta riavviare prima del corso (~30 sec).

# Note didattiche

Il system prompt è volutamente **minimale** e **trasparente**: dice solo al
modello di rispondere e di non fingere di avere strumenti. Non gli chiede di
"essere utile", non gli dà istruzioni elaborate. Questo è il cuore della demo:
mostrare cosa fa un modello quando non è "imbottito" di scaffolding nascosti.

In aula, mostra il system prompt aprendo il pannello "Configurazione" — è
visibile per tutti.

# Verifica funzionamento via curl (opzionale)

```bash
# Health check (sempre aperto, anche con auth attiva)
curl https://<nome-app>.azurewebsites.net/api/health

# Config (richiede token se ACCESS_TOKEN settato)
curl -H "Authorization: Bearer <ACCESS_TOKEN>" \
  https://<nome-app>.azurewebsites.net/api/config

# Chat (richiede token + payload)
curl -X POST https://<nome-app>.azurewebsites.net/api/chat \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Quante R in STRAWBERRY?"}]}'
```
