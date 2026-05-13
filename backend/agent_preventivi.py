"""
Agente Preventivi — conversazione AI multi-turno + costificazione deterministica.

Pattern ibrido:
- Fase conversazionale: Responses API con file_search su baldan-preventivi-kb
- Costificazione: Python puro deterministico (calcola_preventivo)
- PDF: reportlab, stile coerente con agent_bilancio
"""

import json
import os
import re
import shutil
import uuid
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from openai import AzureOpenAI, OpenAIError
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_RESPONSES", "2025-04-01-preview")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
VECTOR_STORE_ID = os.getenv("AZURE_PREVENTIVI_VECTOR_STORE_ID", "")

SESSION_TTL = 1800  # 30 minuti
SESSION_BASE = "/tmp/preventivi_sessions"

# Palette coerente con agent_bilancio
C_DARK = "#1a3a52"
C_MID = "#2d6a8f"
C_RED = "#a02020"
C_LIGHT = "#e8f0f5"

# ---------------------------------------------------------------------------
# Catalogo dati (caricato una volta all'import)
# ---------------------------------------------------------------------------

_CATALOG_DIR = Path(__file__).parent.parent / "data" / "preventivi_catalog"


def _load_catalog(name: str) -> dict:
    with open(_CATALOG_DIR / name, encoding="utf-8") as f:
        return json.load(f)


try:
    MATERIALI = _load_catalog("materiali.json")
    MACCHINE = _load_catalog("macchine.json")
    TEMPI_CICLO = _load_catalog("tempi_ciclo.json")
    COSTI_STAMPI = _load_catalog("costi_stampi.json")
    REGOLE_PRICING = _load_catalog("regole_pricing.json")
except FileNotFoundError as e:
    warnings.warn(f"Catalogo preventivi non trovato: {e}. Alcuni endpoint non funzioneranno.")
    MATERIALI = MACCHINE = TEMPI_CICLO = COSTI_STAMPI = REGOLE_PRICING = {}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
## RUOLO E IDENTITÀ

Sei l'agente preventivi di Baldan Plastica SRL, azienda mantovana di stampaggio plastica a iniezione attiva dal 1965. Aiuti i commerciali a costruire offerte tecnico-economiche per richieste cliente.

Il tuo tono è quello di un commerciale senior con 15 anni di esperienza in azienda: cortese, competente, diretto. Dai del Lei nel primo messaggio, poi adatti al tono del tuo interlocutore. Non sei una segretaria che raccoglie dati — sei un consulente che ragiona insieme al cliente.

Parli sempre in italiano. Niente anglicismi inutili (usa "stampaggio" non "molding", "pezzo" non "part").

---

## OBIETTIVO DELLA CONVERSAZIONE

Raccogliere abbastanza informazioni tecniche da poter costruire un preventivo realistico per uno stampaggio plastica a iniezione. La conversazione è multi-turno: fai 1-3 domande per volta, non bombardare l'utente con questionari. Procedi finché non hai abbastanza dati per la costificazione.

Quando hai tutti i requisiti obbligatori raccolti, segnali che sei pronto a generare il preventivo e attendi conferma esplicita prima di "passare la palla".

---

## REQUISITI DA RACCOGLIERE

### Obbligatori (necessari per costificare)

- tipologia_pezzo: Descrizione breve del pezzo (cosa è, a cosa serve). Es: "coperchio scatola elettronica"
- materiale_codice: Codice del materiale plastico. Vedi catalogo materiali. Se l'utente è vago proponi opzioni in base all'applicazione
- peso_pezzo_g: Peso unitario stimato in grammi. Se l'utente non lo sa, chiedi dimensioni e proponi una stima
- dimensione_max_mm: Lato maggiore del pezzo in mm. Serve per scelta macchina
- complessita: "semplice" / "media" / "complessa"
- volume_annuo_pz: Volume produttivo annuo in pezzi

### Opzionali (utili ma non bloccanti)

- tolleranze: "standard" / "strette" (sotto ±0.1mm)
- finitura_estetica: "tecnica" / "estetica" / "alta estetica"
- lavorazioni_post: Assemblaggio, decorazione, sovrastampaggio, etc.
- urgenza_giorni: Giorni per prima consegna (se urgente)
- certificazioni_richieste: ISO 13485 (medicale), IATF 16949 (automotive), uso alimentare, ecc.
- note_cliente: Qualsiasi info aggiuntiva fornita dal cliente

Regola di pulizia: non chiedere mai contemporaneamente più di 3 campi.

---

## CONOSCENZA TECNICA EMBEDDED

### Warning automatici da segnalare nel campo technical_advice

1. PA66 con spessore < 2mm → "Lo spessore è critico per ritiro e deformazioni con PA66. Valuterei o aumentare lo spessore a 2mm minimo, o passare al PA66-GF30, o al PC per applicazioni meno strutturali."

2. PA o PA66-GF30 senza menzione di stoccaggio controllato → "Questi materiali sono igroscopici, vanno essiccati prima dello stampaggio. Comportano una piccola maggiorazione (~5%) per la gestione."

3. Trasparenza richiesta → "Per la trasparenza l'opzione naturale è il PC, ottica e impatto eccellenti. Se serve solo traslucidità, anche il PP cristallino è un'opzione più economica."

4. Spessore > 4mm o pezzo molto massiccio → "Tempi ciclo lunghi e rischio di difetti di raffreddamento (risucchi, cavità). Valutare alleggerimenti con nervature."

5. Volume < 5.000 pz/anno con stampo complesso → "Su questi volumi il costo stampo è dominante. Considererei due alternative: (a) stampo prototipale meno duraturo a costo ridotto, oppure (b) pensare a una versione multi-cliente del pezzo."

6. Volume > 100.000 pz/anno con stampo mono-impronta → "Su questo volume si giustificherebbe uno stampo a 2 o 4 impronte: investimento iniziale maggiore ma costo pezzo decisamente inferiore."

7. Applicazione medicale o alimentare → "Per questa applicazione servirebbero certificazioni specifiche (ISO 13485 per medicale, dichiarazione di idoneità MOCA per food contact). C'è una piccola maggiorazione del processo."

8. Cliente automotive senza menzione IATF → "Per l'automotive di solito i clienti chiedono certificazione IATF 16949. Lo conferma?"

### Stima della complessità

- semplice: geometrie regolari, nessun sottosquadro, estrazione lineare
- media: dettagli funzionali, fori, nervature, eventuali inserti filettati, sottosquadri minori
- complessa: sottosquadri multipli, filettature interne, sovrastampaggio bi-componente, tolleranze strette

---

## USO DELLA KNOWLEDGE BASE (file_search)

Hai accesso al tool file_search su un vector store dedicato contenente la documentazione tecnica interna Baldan: materiali approvati (ST-MAT-01) e istruzione operativa stampaggio iniezione (IO-PROD-07).

Usalo quando: il cliente chiede se Baldan lavora un materiale specifico, chiede di certificazioni o procedure di qualità, menziona requisiti speciali coperti dalle procedure interne.
Non usarlo per: domande sui prezzi, tempi macchina, raccolta requisiti generica.

Cita sempre la fonte se hai usato file_search: "Secondo la nostra istruzione operativa IO-PROD-07, [...]".

---

## FORMATO DI OUTPUT (JSON OBBLIGATORIO)

A ogni turno produci ESCLUSIVAMENTE un oggetto JSON valido (no markdown, no commenti, no testo prima o dopo):

{
  "reply": "<la tua risposta naturale all'utente, in italiano>",
  "requirements_update": {
    "<campo>": "<nuovo valore>"
  },
  "ready_to_generate": false,
  "technical_advice": null
}

Regole:
- reply: la frase che l'utente vedrà nella chat. Mai più lunga di 3-4 frasi.
- requirements_update: SOLO i campi nuovi o aggiornati in questo turno. Se nessun campo cambia, usa {}.
- ready_to_generate: true solo quando hai TUTTI i requisiti obbligatori E l'utente ha confermato di voler procedere.
- technical_advice: stringa con warning/suggerimento tecnico se applicabile. null se non c'è niente.

---

## QUANDO SEI PRONTO A GENERARE

Quando hai tutti i campi obbligatori, segnala chiaramente nella reply:
"Ho raccolto tutte le informazioni necessarie. Riepilogando: [breve elenco]. Procedo con la costificazione?"

E imposti ready_to_generate: true.\
"""

_COMMENT_SYSTEM = """\
Sei un commerciale senior di Baldan Plastica. Devi scrivere un breve testo
introduttivo (3-4 frasi, massimo 80 parole) da inserire all'inizio di un
preventivo tecnico. Tono professionale ma cordiale, in italiano. Menziona
specificamente il pezzo richiesto e l'applicazione, ringraziando per la
richiesta e introducendo l'offerta. Non inventare dettagli non presenti nei
dati. Restituisci esclusivamente il testo, senza markdown né JSON.\
"""

_COMMENT_FALLBACK = (
    "Vi ringraziamo per la richiesta. "
    "Trovate di seguito la nostra offerta tecnico-economica per la fornitura richiesta."
)

# ---------------------------------------------------------------------------
# Sessioni e client
# ---------------------------------------------------------------------------

SESSIONS: dict[str, dict] = {}
_client: Optional[AzureOpenAI] = None

REQUIRED_FIELDS = [
    "tipologia_pezzo", "materiale_codice", "peso_pezzo_g",
    "dimensione_max_mm", "complessita", "volume_annuo_pz",
]


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not ENDPOINT or not API_KEY:
            raise HTTPException(status_code=500, detail="Azure OpenAI non configurato.")
        _client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
        )
    return _client


def _cleanup_old_sessions() -> None:
    now = datetime.now()
    expired = [
        sid for sid, s in list(SESSIONS.items())
        if (now - s["created_at"]).total_seconds() > SESSION_TTL
    ]
    for sid in expired:
        shutil.rmtree(f"{SESSION_BASE}/{sid}", ignore_errors=True)
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Sessione non trovata o scaduta.")
    return SESSIONS[session_id]


def _session_dir(session_id: str) -> Path:
    return Path(f"{SESSION_BASE}/{session_id}")


def _requirements_completi(req: dict) -> bool:
    return all(req.get(f) is not None for f in REQUIRED_FIELDS)


def _empty_requirements() -> dict:
    return {
        "tipologia_pezzo": None,
        "materiale_codice": None,
        "peso_pezzo_g": None,
        "dimensione_max_mm": None,
        "complessita": None,
        "volume_annuo_pz": None,
        "tolleranze": None,
        "finitura_estetica": None,
        "lavorazioni_post": None,
        "urgenza_giorni": None,
        "certificazioni_richieste": None,
        "note_cliente": None,
    }

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/preventivi", tags=["preventivi"])

# ---------------------------------------------------------------------------
# Parsing JSON
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)

# ---------------------------------------------------------------------------
# Logica catalogo
# ---------------------------------------------------------------------------


def _find_material(codice: str) -> dict:
    for m in MATERIALI.get("materiali", []):
        if m["codice"] == codice:
            return m
    raise ValueError(f"Materiale '{codice}' non trovato nel catalogo.")


def _select_machine(peso_g: float, dimensione_mm: float, materiale_codice: str) -> dict:
    candidates = [
        m for m in MACCHINE.get("macchine", [])
        if m["peso_pezzo_range_g"][0] <= peso_g <= m["peso_pezzo_range_g"][1]
        and m["dimensione_max_pezzo_mm"] >= dimensione_mm
    ]
    if not candidates:
        raise ValueError(
            f"Nessuna macchina adatta per peso {peso_g}g e dimensione {dimensione_mm}mm. "
            "Il pezzo potrebbe essere fuori dai range produttivi."
        )

    candidates.sort(key=lambda m: m["tonnellaggio"])
    base = candidates[0]

    # Correzione tonnellaggio per materiali rinforzati
    mult = 1.10 if "-GF" in materiale_codice else 1.0
    min_ton = base["tonnellaggio"] * mult

    for m in candidates:
        if m["tonnellaggio"] >= min_ton:
            return m

    raise ValueError(
        f"Nessuna macchina con tonnellaggio sufficiente (≥{min_ton:.0f}t) "
        f"per il materiale {materiale_codice} con questo pezzo."
    )


def _find_cycle_time_base(complessita: str, peso_g: float) -> int:
    matrice = TEMPI_CICLO.get("matrice_tempi", {}).get("complessita", {})
    tempi = matrice.get(complessita, {}).get("tempi_per_peso_g", {})
    if not tempi:
        raise ValueError(f"Complessità '{complessita}' non trovata nel catalogo tempi ciclo.")

    for key, val in tempi.items():
        lo, hi = map(int, key.split("-"))
        if lo <= peso_g < hi:
            return val
    # oltre il range massimo: ultimo bucket
    return list(tempi.values())[-1]


def _apply_cycle_corrections(
    base_sec: int, materiale_codice: str, tolleranze: Optional[str]
) -> tuple[float, list[dict]]:
    factors = []
    result = float(base_sec)

    if "-GF" in materiale_codice:
        factors.append({"nome": "Materiale rinforzato fibra vetro", "moltiplicatore": 1.10})
        result *= 1.10

    if materiale_codice.startswith("PA") or materiale_codice == "POM":
        factors.append({"nome": "Materiale con ritiro elevato (PA/POM)", "moltiplicatore": 1.15})
        result *= 1.15

    if tolleranze == "strette":
        factors.append({"nome": "Tolleranze critiche", "moltiplicatore": 1.20})
        result *= 1.20

    return round(result, 2), factors


def calcola_preventivo(requirements: dict) -> dict:
    peso_g = float(requirements["peso_pezzo_g"])
    dimensione_mm = float(requirements["dimensione_max_mm"])
    materiale_codice = str(requirements["materiale_codice"])
    complessita = str(requirements["complessita"])
    volume_annuo = int(requirements["volume_annuo_pz"])
    tolleranze = requirements.get("tolleranze")
    certificazioni = requirements.get("certificazioni_richieste") or ""
    urgenza_giorni = requirements.get("urgenza_giorni")

    # 6.1 Materiale
    mat = _find_material(materiale_codice)
    prezzo_kg = mat["prezzo_kg"]

    # 6.2 Macchina
    macchina = _select_machine(peso_g, dimensione_mm, materiale_codice)

    # 6.3 Tempo ciclo
    tempo_base = _find_cycle_time_base(complessita, peso_g)
    tempo_finale, fattori = _apply_cycle_corrections(tempo_base, materiale_codice, tolleranze)

    # 6.4 Calcolo costo pezzo
    costo_mat = round((peso_g / 1000) * prezzo_kg, 4)
    scarto = round(costo_mat * REGOLE_PRICING.get("scarto_produttivo", {}).get("valore_percentuale", 0.05), 4)
    costo_macchina = round((tempo_finale / 3600) * macchina["costo_orario"], 4)
    costo_industriale = round(costo_mat + scarto + costo_macchina, 4)

    markup_pct = REGOLE_PRICING.get("markup", {}).get("valore_default", 0.22)
    markup_eur = round(costo_industriale * markup_pct, 4)
    subtotale = round(costo_industriale + markup_eur, 4)

    # 6.5 Sconto volume
    sconto_pct = 0.0
    sconto_label = "Standard"
    for fascia in REGOLE_PRICING.get("sconti_volume", {}).get("fasce", []):
        if fascia["volume_annuo_da"] <= volume_annuo < fascia["volume_annuo_a"]:
            sconto_pct = fascia["sconto"]
            sconto_label = fascia["label"]
            break
    sconto_eur = round(-subtotale * sconto_pct, 4)

    # 6.6 Maggiorazioni
    maggiorazioni = []
    magg_cfg = REGOLE_PRICING.get("maggiorazioni", {})

    if materiale_codice.startswith("PA") or "-GF" in materiale_codice:
        pct = magg_cfg.get("materiale_critico", {}).get("valore_percentuale", 0.05)
        maggiorazioni.append({
            "nome": "Materiale critico (PA/GF)",
            "pct": pct,
            "eur": round(subtotale * pct, 4),
        })

    cert_lower = certificazioni.lower()
    if any(k in cert_lower for k in ["iatf", "iso 13485", "moca", "alimentare"]):
        pct = magg_cfg.get("qualita_certificata", {}).get("valore_percentuale", 0.08)
        maggiorazioni.append({
            "nome": "Certificazione speciale (IATF/ISO 13485/MOCA)",
            "pct": pct,
            "eur": round(subtotale * pct, 4),
        })

    if urgenza_giorni is not None:
        urg = magg_cfg.get("urgenza_consegna", {})
        if urgenza_giorni < 15:
            pct = urg.get("consegna_sotto_15_giorni", 0.20)
        elif urgenza_giorni < 30:
            pct = urg.get("consegna_15_30_giorni", 0.12)
        elif urgenza_giorni <= 60:
            pct = urg.get("consegna_30_60_giorni", 0.05)
        else:
            pct = 0.0
        if pct > 0:
            maggiorazioni.append({
                "nome": f"Urgenza consegna ({urgenza_giorni} giorni)",
                "pct": pct,
                "eur": round(subtotale * pct, 4),
            })

    # 6.7 Prezzo finale
    prezzo_finale = round(
        subtotale + sconto_eur + sum(m["eur"] for m in maggiorazioni), 4
    )

    # 6.8 Stampo
    if dimensione_mm < 100:
        dim_cat = "piccolo"
    elif dimensione_mm <= 300:
        dim_cat = "medio"
    else:
        dim_cat = "grande"
    stampo_key = f"{complessita}_{dim_cat}"
    stampo_data = COSTI_STAMPI.get("fasce_costo_stampo", {}).get(stampo_key, {})
    tempi_s = stampo_data.get("tempo_realizzazione_settimane", [0, 0])

    # 6.9 Ricavi annui
    ricavi_annui = round(volume_annuo * prezzo_finale, 2)

    condizioni = REGOLE_PRICING.get("condizioni_commerciali_standard", {})
    esclusioni = REGOLE_PRICING.get("esclusioni_standard", [])

    return {
        "azienda_offerente": "Baldan Plastica SRL",
        "data_offerta": date.today().isoformat(),
        "validita_giorni": condizioni.get("validita_offerta_giorni", 30),
        "requisiti": dict(requirements),
        "selezione_macchina": {
            "codice": macchina["codice"],
            "tonnellaggio": macchina["tonnellaggio"],
            "costo_orario_eur": macchina["costo_orario"],
            "motivazione": "Selezionata in base a peso pezzo e dimensione massima",
        },
        "tempo_ciclo": {
            "tempo_base_sec": tempo_base,
            "fattori_correttivi": fattori,
            "tempo_finale_sec": tempo_finale,
        },
        "costo_pezzo": {
            "costo_materiale_eur": costo_mat,
            "scarto_eur": scarto,
            "costo_macchina_eur": costo_macchina,
            "costo_industriale_eur": costo_industriale,
            "markup_pct": markup_pct,
            "markup_eur": markup_eur,
            "subtotale_eur": subtotale,
            "sconto_volume": {
                "fascia": sconto_label,
                "pct": sconto_pct,
                "eur": sconto_eur,
            },
            "maggiorazioni": maggiorazioni,
            "prezzo_unitario_finale_eur": prezzo_finale,
        },
        "stampo": {
            "categoria": stampo_key,
            "range_eur": stampo_data.get("range_eur", [0, 0]),
            "valore_medio_eur": stampo_data.get("valore_medio", 0),
            "numero_impronte": 1,
            "tempo_realizzazione_settimane": f"{tempi_s[0]}-{tempi_s[1]}",
        },
        "ricavi_annui_stimati_eur": ricavi_annui,
        "condizioni_commerciali": condizioni,
        "esclusioni": esclusioni,
    }

# ---------------------------------------------------------------------------
# Chiamate AI
# ---------------------------------------------------------------------------


def _extract_output_text(response) -> str:
    return "".join(
        block.text
        for item in response.output if item.type == "message"
        for block in item.content if block.type == "output_text"
    )


def _call_agent(client: AzureOpenAI, messages: list, extra_prompt: str = "") -> dict:
    input_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    if extra_prompt:
        input_msgs[-1]["content"] = extra_prompt + "\n\n" + input_msgs[-1]["content"]

    kwargs = dict(
        model=DEPLOYMENT,
        instructions=_AGENT_SYSTEM_PROMPT,
        input=input_msgs,
    )
    if VECTOR_STORE_ID:
        kwargs["tools"] = [{"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}]
        kwargs["include"] = ["file_search_call.results"]

    resp = client.responses.create(**kwargs)
    raw = _extract_output_text(resp)
    return _parse_json_response(raw)


def _call_comment(client: AzureOpenAI, preventivo_data: dict) -> str:
    req = preventivo_data.get("requisiti", {})
    user_text = (
        f"Dati del preventivo:\n"
        f"- Pezzo: {req.get('tipologia_pezzo', 'n.d.')}\n"
        f"- Materiale: {req.get('materiale_codice', 'n.d.')}\n"
        f"- Volume annuo: {req.get('volume_annuo_pz', 'n.d.')} pezzi\n"
        f"- Note cliente: {req.get('note_cliente') or 'nessuna nota'}\n\n"
        "Scrivi il testo introduttivo."
    )
    try:
        resp = client.responses.create(
            model=DEPLOYMENT,
            instructions=_COMMENT_SYSTEM,
            input=[{"role": "user", "content": user_text}],
        )
        return _extract_output_text(resp).strip()
    except OpenAIError:
        return _COMMENT_FALLBACK

# ---------------------------------------------------------------------------
# Generazione PDF
# ---------------------------------------------------------------------------


def _fmt_eur(v, decimals=4):
    if v is None:
        return "n.d."
    return f"€ {v:,.{decimals}f}"


def _build_pdf(session_id: str, preventivo_data: dict) -> str:
    outdir = _session_dir(session_id)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = str(outdir / "offerta.pdf")

    req = preventivo_data.get("requisiti", {})
    cp = preventivo_data.get("costo_pezzo", {})
    stampo = preventivo_data.get("stampo", {})
    macchina = preventivo_data.get("selezione_macchina", {})
    tc = preventivo_data.get("tempo_ciclo", {})
    condizioni = preventivo_data.get("condizioni_commerciali", {})
    esclusioni = preventivo_data.get("esclusioni", [])

    doc = SimpleDocTemplate(
        outpath,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    base = getSampleStyleSheet()
    s_title = ParagraphStyle("Title2", parent=base["Title"], textColor=colors.HexColor(C_DARK),
                              fontSize=22, spaceAfter=10)
    s_sub = ParagraphStyle("Sub2", parent=base["Normal"], textColor=colors.HexColor(C_MID),
                            fontSize=13, spaceAfter=6)
    s_h2 = ParagraphStyle("H22", parent=base["Heading2"], textColor=colors.HexColor(C_DARK),
                           fontSize=14, spaceBefore=14, spaceAfter=6)
    s_body = ParagraphStyle("Body2", parent=base["Normal"], fontSize=10, leading=15, spaceAfter=4)
    s_bullet = ParagraphStyle("Bullet2", parent=base["Normal"], fontSize=10, leading=14,
                               leftIndent=15, spaceAfter=3)
    s_footer = ParagraphStyle("Footer2", parent=base["Normal"], fontSize=8,
                               textColor=colors.grey, alignment=TA_CENTER)

    story = []

    # ---- Pagina 1: Copertina ----
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Offerta Tecnico-Economica", s_title))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(str(req.get("tipologia_pezzo", "—")), s_sub))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Offerente: Baldan Plastica SRL", s_body))
    story.append(Paragraph("Cliente: [da definire]", s_body))
    story.append(Paragraph(f"Data offerta: {preventivo_data.get('data_offerta', '—')}", s_body))
    story.append(Paragraph(f"Validità: {preventivo_data.get('validita_giorni', 30)} giorni", s_body))
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph(
        "Documento generato automaticamente. I valori sono indicativi e da confermare "
        "sulla base del disegno tecnico definitivo.",
        s_footer,
    ))
    story.append(PageBreak())

    # ---- Pagina 2: Commento + Specifiche ----
    story.append(Paragraph("Presentazione dell'Offerta", s_h2))
    commento = preventivo_data.get("commento_commerciale", _COMMENT_FALLBACK)
    box_table = Table(
        [[Paragraph(commento, s_body)]],
        colWidths=[doc.width],
    )
    box_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(C_LIGHT)),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(C_MID)),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(box_table)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Specifiche tecniche del pezzo", s_h2))
    req_labels = [
        ("Tipologia pezzo", "tipologia_pezzo"),
        ("Materiale plastico", "materiale_codice"),
        ("Peso unitario", "peso_pezzo_g", lambda v: f"{v} g"),
        ("Dimensione massima", "dimensione_max_mm", lambda v: f"{v} mm"),
        ("Complessità geometrica", "complessita"),
        ("Volume annuo previsto", "volume_annuo_pz", lambda v: f"{int(v):,} pz/anno"),
        ("Tolleranze", "tolleranze"),
        ("Finitura estetica", "finitura_estetica"),
        ("Lavorazioni post-processo", "lavorazioni_post"),
        ("Urgenza consegna", "urgenza_giorni", lambda v: f"{v} giorni"),
        ("Certificazioni richieste", "certificazioni_richieste"),
        ("Note cliente", "note_cliente"),
    ]
    req_rows = []
    for item in req_labels:
        label = item[0]
        key = item[1]
        fmt = item[2] if len(item) > 2 else None
        val = req.get(key)
        if val is not None:
            display = fmt(val) if fmt else str(val)
            req_rows.append([label, display])

    if req_rows:
        req_table = Table(req_rows, colWidths=[doc.width * 0.45, doc.width * 0.55])
        req_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor(C_LIGHT)]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(req_table)
    story.append(PageBreak())

    # ---- Pagina 3: Dettaglio costi ----
    story.append(Paragraph("Dettaglio Costificazione", s_h2))

    cost_rows = [
        ["Voce", "Importo (€/pz)"],
        ["Costo materiale", f"€ {cp.get('costo_materiale_eur', 0):.4f}"],
        ["Scarto produttivo (5%)", f"€ {cp.get('scarto_eur', 0):.4f}"],
        ["Costo lavorazione macchina", f"€ {cp.get('costo_macchina_eur', 0):.4f}"],
        ["COSTO INDUSTRIALE", f"€ {cp.get('costo_industriale_eur', 0):.4f}"],
        [f"Margine commerciale ({cp.get('markup_pct', 0.22):.0%})", f"€ {cp.get('markup_eur', 0):.4f}"],
        ["SUBTOTALE", f"€ {cp.get('subtotale_eur', 0):.4f}"],
    ]

    sv = cp.get("sconto_volume", {})
    if sv.get("pct", 0) > 0:
        cost_rows.append([
            f"Sconto volume ({sv.get('fascia', '')} — {sv.get('pct', 0):.0%})",
            f"€ {sv.get('eur', 0):.4f}",
        ])

    for m in cp.get("maggiorazioni", []):
        cost_rows.append([
            f"Maggiorazione: {m['nome']} (+{m['pct']:.0%})",
            f"€ {m['eur']:.4f}",
        ])

    cost_rows.append(["PREZZO UNITARIO DI VENDITA", f"€ {cp.get('prezzo_unitario_finale_eur', 0):.4f}"])

    col_w = [doc.width * 0.65, doc.width * 0.35]
    cost_table = Table(cost_rows, colWidths=col_w, repeatRows=1)

    bold_rows = [i for i, r in enumerate(cost_rows)
                 if r[0] in ("COSTO INDUSTRIALE", "SUBTOTALE", "PREZZO UNITARIO DI VENDITA")]
    last_row = len(cost_rows) - 1

    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_DARK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor(C_LIGHT)]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        # Final row highlight
        ("BACKGROUND", (0, last_row), (-1, last_row), colors.HexColor(C_DARK)),
        ("TEXTCOLOR", (0, last_row), (-1, last_row), colors.white),
        ("FONTNAME", (0, last_row), (-1, last_row), "Helvetica-Bold"),
    ])
    for r in bold_rows:
        ts.add("FONTNAME", (0, r), (-1, r), "Helvetica-Bold")
    # Sconto volume in rosso
    for i, row in enumerate(cost_rows):
        if "Sconto" in row[0]:
            ts.add("TEXTCOLOR", (1, i), (1, i), colors.HexColor(C_RED))

    cost_table.setStyle(ts)
    story.append(cost_table)

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Costo Stampo", s_h2))

    rang = stampo.get("range_eur", [0, 0])
    stampo_text = (
        f"Categoria: {stampo.get('categoria', '—')}  |  "
        f"Range stimato: € {rang[0]:,} — € {rang[1]:,}  |  "
        f"Valore medio: € {stampo.get('valore_medio_eur', 0):,}  |  "
        f"Tempo realizzazione: {stampo.get('tempo_realizzazione_settimane', '—')} settimane"
    )
    stampo_box = Table([[Paragraph(stampo_text, s_body)]], colWidths=[doc.width])
    stampo_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(C_LIGHT)),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(C_MID)),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(stampo_box)

    story.append(Spacer(1, 0.4 * cm))
    ricavi = preventivo_data.get("ricavi_annui_stimati_eur", 0)
    story.append(Paragraph(
        f"<b>Ricavi annui stimati:</b> € {ricavi:,.2f}  "
        f"({req.get('volume_annuo_pz', 0):,} pz × € {cp.get('prezzo_unitario_finale_eur', 0):.4f}/pz)",
        s_body,
    ))
    story.append(PageBreak())

    # ---- Pagina 4: Condizioni commerciali ----
    story.append(Paragraph("Condizioni Commerciali", s_h2))
    tempi_stampo = stampo.get("tempo_realizzazione_settimane", "—")
    cond_rows = [
        ["Validità offerta", f"{condizioni.get('validita_offerta_giorni', 30)} giorni"],
        ["Termini di pagamento", condizioni.get("termini_pagamento", "—")],
        ["Modalità di pagamento", condizioni.get("modalita_pagamento", "—")],
        ["Tempi consegna stampo", f"{tempi_stampo} settimane dal collaudo"],
        ["Prima serie", condizioni.get("tempo_consegna_prima_serie", "—")],
        ["Serie successive", condizioni.get("tempo_consegna_serie_successive", "—")],
        ["Garanzia", condizioni.get("garanzia_pezzo", "—")],
        ["Imballaggio", condizioni.get("imballaggio", "—")],
        ["Luogo di resa", condizioni.get("luogo_resa", "—")],
    ]
    cond_table = Table(cond_rows, colWidths=[doc.width * 0.35, doc.width * 0.65])
    cond_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor(C_LIGHT)]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(cond_table)

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Esclusioni", s_h2))
    for exc in esclusioni:
        story.append(Paragraph(f"• {exc}", s_bullet))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "Documento generato automaticamente da un agente AI per scopi dimostrativi. "
        "I prezzi e i tempi sono indicativi e devono essere validati da un commerciale "
        "prima di costituire offerta formale.",
        s_footer,
    ))

    doc.build(story)
    return outpath

# ---------------------------------------------------------------------------
# Modelli request/response
# ---------------------------------------------------------------------------


class MessageRequest(BaseModel):
    session_id: str
    user_message: str


class GenerateRequest(BaseModel):
    session_id: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_OPENING_MESSAGE = (
    "Buongiorno, sono l'agente preventivi di Baldan Plastica. "
    "Sono qui per aiutarLa a costruire una prima quotazione per la Sua richiesta. "
    "Mi può descrivere brevemente il pezzo che il Vostro cliente Vi ha chiesto?"
)


@router.post("/start")
def start_session():
    _cleanup_old_sessions()
    session_id = str(uuid.uuid4())
    _session_dir(session_id).mkdir(parents=True, exist_ok=True)

    SESSIONS[session_id] = {
        "created_at": datetime.now(),
        "messages": [{"role": "assistant", "content": _OPENING_MESSAGE}],
        "requirements": _empty_requirements(),
        "ready_to_generate": False,
        "preventivo_data": None,
        "report_path": None,
        "advice_list": [],
    }
    return {"session_id": session_id, "opening_message": _OPENING_MESSAGE}


@router.post("/message")
def message(body: MessageRequest):
    _cleanup_old_sessions()
    session = _get_session(body.session_id)
    client = _get_client()

    session["messages"].append({"role": "user", "content": body.user_message})

    try:
        parsed = _call_agent(client, session["messages"])
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = _call_agent(
                client, session["messages"],
                "IMPORTANTE: rispondi SOLO con JSON puro, nessun testo o markdown.",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            session["messages"].pop()
            raise HTTPException(
                status_code=500,
                detail=f"Il modello non ha prodotto JSON valido: {exc}",
            ) from exc
    except OpenAIError as exc:
        session["messages"].pop()
        raise HTTPException(status_code=502, detail=f"Errore AI: {exc}") from exc

    reply = parsed.get("reply", "")
    req_update = parsed.get("requirements_update") or {}
    technical_advice = parsed.get("technical_advice")

    # Merge requirements (non-None sovrascrivono)
    for k, v in req_update.items():
        if v is not None and k in session["requirements"]:
            session["requirements"][k] = v

    # Accumula advice per il PDF
    if technical_advice:
        session["advice_list"].append(technical_advice)

    # Valida ready_to_generate lato server
    ready = bool(parsed.get("ready_to_generate", False)) and _requirements_completi(session["requirements"])
    session["ready_to_generate"] = ready

    session["messages"].append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "requirements": session["requirements"],
        "ready_to_generate": ready,
        "technical_advice": technical_advice,
    }


@router.post("/generate")
def generate(body: GenerateRequest):
    _cleanup_old_sessions()
    session = _get_session(body.session_id)

    if not _requirements_completi(session["requirements"]):
        missing = [f for f in REQUIRED_FIELDS if session["requirements"].get(f) is None]
        raise HTTPException(
            status_code=400,
            detail=f"Requisiti obbligatori mancanti: {', '.join(missing)}",
        )

    client = _get_client()

    try:
        preventivo_data = calcola_preventivo(session["requirements"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    commento = _call_comment(client, preventivo_data)
    preventivo_data["commento_commerciale"] = commento
    if session["advice_list"]:
        preventivo_data["suggerimenti_tecnici"] = session["advice_list"]

    try:
        report_path = _build_pdf(body.session_id, preventivo_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore generazione PDF: {exc}") from exc

    session["preventivo_data"] = preventivo_data
    session["report_path"] = report_path

    return {
        "status": "completed",
        "preventivo_data": preventivo_data,
        "download_url": f"/api/preventivi/download/{body.session_id}",
    }


@router.get("/preventivo/{session_id}")
def get_preventivo(session_id: str):
    _cleanup_old_sessions()
    session = _get_session(session_id)
    if not session.get("preventivo_data"):
        raise HTTPException(status_code=404, detail="Preventivo non ancora generato.")
    return session["preventivo_data"]


@router.get("/download/{session_id}")
def download_pdf(session_id: str):
    _cleanup_old_sessions()
    session = _get_session(session_id)
    if not session.get("report_path") or not Path(session["report_path"]).exists():
        raise HTTPException(status_code=404, detail="PDF non ancora generato.")

    tipologia = (
        session.get("preventivo_data", {}).get("requisiti", {}).get("tipologia_pezzo", "offerta")
        or "offerta"
    )
    slug = re.sub(r"[^a-z0-9]+", "_", tipologia.lower())[:30].strip("_")
    filename = f"offerta_{slug}.pdf"

    return FileResponse(
        session["report_path"],
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
