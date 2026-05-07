"""
Agente Bilancio — workflow deterministico a 4 step su PDF di bilancio.

Architettura: il backend orchestra 4 step fissi in ordine fisso.
L'AI viene chiamata solo per estrazione dati (step 1) e commento (step 4).
Step 2 e 3 sono puro codice Python deterministico.

Sessioni in-memory: niente DB, TTL 30 minuti, file temporanei in /tmp.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend non-interattivo — obbligatorio su server
import matplotlib.pyplot as plt
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import AzureOpenAI, OpenAIError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_RESPONSES", "2025-04-01-preview")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

SESSION_TTL = 1800  # 30 minuti
MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB

# Palette grafici e report
C_DARK = "#1a3a52"
C_MID = "#2d6a8f"
C_RED = "#a02020"
C_LIGHT = "#e8f0f5"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_STEP1_SYSTEM = """\
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
Se un dato non è disponibile nel PDF, usa null (non zero).\
"""

_STEP4_SYSTEM = """\
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
}\
"""

# ---------------------------------------------------------------------------
# Sessioni e client
# ---------------------------------------------------------------------------

SESSIONS: dict[str, dict] = {}
_client: Optional[AzureOpenAI] = None


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
        shutil.rmtree(f"/tmp/agent_sessions/{sid}", ignore_errors=True)
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Sessione non trovata o scaduta.")
    return SESSIONS[session_id]


def _session_dir(session_id: str) -> Path:
    return Path(f"/tmp/agent_sessions/{session_id}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/agent", tags=["agent"])

# ---------------------------------------------------------------------------
# Task 2 — Upload
# ---------------------------------------------------------------------------


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    _cleanup_old_sessions()

    content = await file.read()

    if len(content) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="File troppo grande (max 10 MB).")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Il file caricato non è un PDF valido.")

    client = _get_client()
    filename = file.filename or "bilancio.pdf"

    try:
        uploaded = client.files.create(
            file=(filename, content, "application/pdf"),
            purpose="assistants",
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Errore upload: {exc}") from exc

    session_id = str(uuid.uuid4())
    _session_dir(session_id).mkdir(parents=True, exist_ok=True)

    SESSIONS[session_id] = {
        "created_at": datetime.now(),
        "pdf_file_id": uploaded.id,
        "filename": filename,
        "extracted_data": None,
        "kpis": None,
        "chart_paths": [],
        "report_path": None,
    }

    return {"session_id": session_id, "filename": filename}


# ---------------------------------------------------------------------------
# Task 3 — Step 1: estrazione dati AI
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict:
    """Rimuove eventuali markdown fence e parsa il JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def _call_extraction(client: AzureOpenAI, file_id: str, extra: str = "") -> dict:
    user_text = "Estrai i dati di bilancio dal documento allegato."
    if extra:
        user_text = extra + "\n\n" + user_text

    resp = client.responses.create(
        model=DEPLOYMENT,
        instructions=_STEP1_SYSTEM,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": user_text},
            ],
        }],
    )

    raw = "".join(
        block.text
        for item in resp.output if item.type == "message"
        for block in item.content if block.type == "output_text"
    )
    return _parse_json_response(raw)


@router.post("/step1")
def step1_extract(session_id: str):
    _cleanup_old_sessions()
    session = _get_session(session_id)
    client = _get_client()
    file_id = session["pdf_file_id"]

    try:
        data = _call_extraction(client, file_id)
    except (json.JSONDecodeError, ValueError):
        try:
            data = _call_extraction(
                client, file_id,
                "IMPORTANTE: rispondi SOLO con JSON puro, nessun testo o markdown.",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Impossibile estrarre JSON valido dal PDF: {exc}",
            ) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Errore AI step 1: {exc}") from exc

    session["extracted_data"] = data

    azienda = data.get("azienda", "—")
    anni = data.get("anni", [])
    anni_str = ", ".join(str(a) for a in anni)
    ce = data.get("conto_economico", {})
    sp = data.get("stato_patrimoniale", {})
    n_ce = len(anni) * len(next(iter(ce.values()), {}).keys()) if ce else 0
    n_sp = len(anni) * len(next(iter(sp.values()), {}).keys()) if sp else 0

    return {
        "step": 1,
        "status": "completed",
        "summary": f"Estratti dati per gli anni {anni_str} — {azienda}",
        "details": [
            f"Conto Economico: {n_ce} voci × {len(anni)} anni",
            f"Stato Patrimoniale: {n_sp} voci × {len(anni)} anni",
        ],
    }


# ---------------------------------------------------------------------------
# Task 4 — Step 2: calcolo KPI
# ---------------------------------------------------------------------------


def _sdiv(num, den, mult: float = 1.0, decimals: int = 2):
    """Safe division — ritorna None se denominatore è 0 o None."""
    if num is None or den is None or den == 0:
        return None
    return round((num / den) * mult, decimals)


def _spct(curr, prev):
    """Variazione percentuale sicura."""
    if curr is None or prev is None or prev == 0:
        return None
    return round((curr - prev) / abs(prev) * 100, 1)


@router.post("/step2")
def step2_kpis(session_id: str):
    _cleanup_old_sessions()
    session = _get_session(session_id)
    if not session["extracted_data"]:
        raise HTTPException(status_code=400, detail="Eseguire prima step 1.")

    data = session["extracted_data"]
    anni = [str(a) for a in sorted(data.get("anni", []))]
    ce = data.get("conto_economico", {})
    sp = data.get("stato_patrimoniale", {})

    kpis: dict = {"anni": anni, "per_anno": {}, "variazioni": {}, "cagr_ricavi": None}

    for a in anni:
        c = ce.get(a, {})
        s = sp.get(a, {})
        ricavi = c.get("ricavi")
        ebitda = c.get("ebitda")
        ebit = c.get("ebit")
        utile = c.get("utile_netto")
        oneri = c.get("oneri_finanziari")
        pfn = s.get("posizione_finanziaria_netta")
        pn = s.get("patrimonio_netto")
        cin = s.get("capitale_investito_netto")

        kpis["per_anno"][a] = {
            "ebitda_margin": _sdiv(ebitda, ricavi, 100),
            "ebit_margin": _sdiv(ebit, ricavi, 100),
            "net_margin": _sdiv(utile, ricavi, 100),
            "pfn_ebitda": _sdiv(pfn, ebitda),
            "pfn_pn": _sdiv(pfn, pn),
            "roi": _sdiv(ebit, cin, 100),
            "roe": _sdiv(utile, pn, 100),
            "copertura_interessi": _sdiv(ebitda, abs(oneri) if oneri else None),
        }

    for i in range(1, len(anni)):
        prev_a, curr_a = anni[i - 1], anni[i]
        cp, cc = ce.get(prev_a, {}), ce.get(curr_a, {})
        label = f"{curr_a[-2:]}/{prev_a[-2:]}"
        kpis["variazioni"][label] = {
            "ricavi": _spct(cc.get("ricavi"), cp.get("ricavi")),
            "ebitda": _spct(cc.get("ebitda"), cp.get("ebitda")),
            "ebit": _spct(cc.get("ebit"), cp.get("ebit")),
            "utile_netto": _spct(cc.get("utile_netto"), cp.get("utile_netto")),
        }

    if len(anni) >= 2:
        r0 = ce.get(anni[0], {}).get("ricavi")
        rn = ce.get(anni[-1], {}).get("ricavi")
        n = int(anni[-1]) - int(anni[0])
        if r0 and rn and r0 > 0 and n > 0:
            kpis["cagr_ricavi"] = round(((rn / r0) ** (1 / n) - 1) * 100, 1)

    session["kpis"] = kpis

    last = anni[-1] if anni else None
    details = []
    if last:
        pa = kpis["per_anno"].get(last, {})
        if pa.get("ebitda_margin") is not None:
            details.append(f"EBITDA margin {last}: {pa['ebitda_margin']}%")
        if pa.get("pfn_ebitda") is not None:
            details.append(f"PFN/EBITDA {last}: {pa['pfn_ebitda']}x")
    if kpis["variazioni"]:
        last_var = list(kpis["variazioni"].values())[-1]
        last_key = list(kpis["variazioni"].keys())[-1]
        v = last_var.get("ricavi")
        if v is not None:
            sign = "+" if v > 0 else ""
            details.append(f"Var. ricavi {last_key}: {sign}{v}%")

    n_kpis = len(anni) * 8 + len(kpis["variazioni"]) * 4
    return {
        "step": 2,
        "status": "completed",
        "summary": f"Calcolati {n_kpis} KPI",
        "details": details,
    }


# ---------------------------------------------------------------------------
# Task 5 — Step 3: generazione grafici
# ---------------------------------------------------------------------------


def _v(d: dict, key: str, default=0):
    val = d.get(key)
    return val if val is not None else default


def _generate_charts(session_id: str, data: dict, kpis: dict) -> list[str]:
    outdir = _session_dir(session_id)
    anni = kpis["anni"]
    ce = data.get("conto_economico", {})
    sp = data.get("stato_patrimoniale", {})
    pa = kpis["per_anno"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.axis": "y",
        "grid.alpha": 0.4,
    })

    paths = []

    # — Grafico 1: Trend Ricavi vs EBITDA —
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(anni))
    width = 0.35
    ricavi = [_v(ce.get(a, {}), "ricavi") / 1000 for a in anni]
    ebitda = [_v(ce.get(a, {}), "ebitda") / 1000 for a in anni]
    bars1 = ax.bar([i - width / 2 for i in x], ricavi, width, label="Ricavi", color=C_DARK)
    bars2 = ax.bar([i + width / 2 for i in x], ebitda, width, label="EBITDA", color=C_MID)
    ax.set_xticks(list(x))
    ax.set_xticklabels(anni)
    ax.set_ylabel("M€")
    ax.set_title("Trend Ricavi vs EBITDA")
    ax.legend()
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    p1 = str(outdir / "trend_ricavi_ebitda.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # — Grafico 2: Marginalità —
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = ["o", "s", "^"]
    series = [
        ("EBITDA margin %", [pa.get(a, {}).get("ebitda_margin") for a in anni], C_DARK),
        ("EBIT margin %", [pa.get(a, {}).get("ebit_margin") for a in anni], C_MID),
        ("Net margin %", [pa.get(a, {}).get("net_margin") for a in anni], C_RED),
    ]
    for (label, vals, col), m in zip(series, markers):
        clean = [v if v is not None else float("nan") for v in vals]
        ax.plot(anni, clean, marker=m, color=col, linewidth=2, label=label)
    ax.set_ylabel("%")
    ax.set_title("Evoluzione Marginalità")
    ax.legend()
    fig.tight_layout()
    p2 = str(outdir / "marginalita.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # — Grafico 3: Struttura finanziaria —
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    x = range(len(anni))
    width = 0.35
    pfn_vals = [_v(sp.get(a, {}), "posizione_finanziaria_netta") / 1000 for a in anni]
    pn_vals = [_v(sp.get(a, {}), "patrimonio_netto") / 1000 for a in anni]
    pfn_ebitda_vals = [pa.get(a, {}).get("pfn_ebitda") for a in anni]

    ax1.bar([i - width / 2 for i in x], pfn_vals, width, label="PFN", color=C_RED, alpha=0.85)
    ax1.bar([i + width / 2 for i in x], pn_vals, width, label="Patrimonio Netto", color=C_DARK, alpha=0.85)
    clean_ratio = [v if v is not None else float("nan") for v in pfn_ebitda_vals]
    ax2.plot(list(x), clean_ratio, marker="D", color=C_MID, linewidth=2, label="PFN/EBITDA (dx)")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(anni)
    ax1.set_ylabel("M€")
    ax2.set_ylabel("Multiplo PFN/EBITDA")
    ax1.set_title("Struttura Finanziaria")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    fig.tight_layout()
    p3 = str(outdir / "struttura_finanziaria.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    return paths


@router.post("/step3")
def step3_charts(session_id: str):
    _cleanup_old_sessions()
    session = _get_session(session_id)
    if not session["kpis"]:
        raise HTTPException(status_code=400, detail="Eseguire prima step 2.")

    paths = _generate_charts(session_id, session["extracted_data"], session["kpis"])
    session["chart_paths"] = paths

    return {
        "step": 3,
        "status": "completed",
        "summary": f"Generati {len(paths)} grafici",
        "details": ["Trend ricavi vs EBITDA", "Marginalita'", "Struttura finanziaria"],
    }


# ---------------------------------------------------------------------------
# Task 6 — Step 4: commento AI + composizione PDF
# ---------------------------------------------------------------------------


def _call_commentary(client: AzureOpenAI, kpis: dict, note: str) -> dict:
    user_text = (
        f"Ecco i dati di bilancio:\nKPI: {json.dumps(kpis, ensure_ascii=False)}\n"
        f"Note qualitative dal bilancio: {note or 'Non disponibili.'}\n\nScrivi il commento."
    )
    resp = client.responses.create(
        model=DEPLOYMENT,
        instructions=_STEP4_SYSTEM,
        input=[{"role": "user", "content": user_text}],
    )
    raw = "".join(
        block.text
        for item in resp.output if item.type == "message"
        for block in item.content if block.type == "output_text"
    )
    return _parse_json_response(raw)


def _fmt(v, suffix="", none_str="n.d."):
    if v is None:
        return none_str
    return f"{v}{suffix}"


def _build_pdf(session_id: str, data: dict, kpis: dict, commentary: dict) -> str:
    outpath = str(_session_dir(session_id) / "report.pdf")
    azienda = data.get("azienda", "—")
    anni = kpis["anni"]
    ce = data.get("conto_economico", {})
    sp = data.get("stato_patrimoniale", {})
    pa = kpis["per_anno"]
    variazioni = kpis.get("variazioni", {})
    chart_paths = _session_dir(session_id)

    doc = SimpleDocTemplate(
        outpath,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    base = getSampleStyleSheet()
    s_title = ParagraphStyle("Title", parent=base["Title"], textColor=colors.HexColor(C_DARK),
                              fontSize=22, spaceAfter=10)
    s_sub = ParagraphStyle("Sub", parent=base["Normal"], textColor=colors.HexColor(C_MID),
                            fontSize=13, spaceAfter=6)
    s_h2 = ParagraphStyle("H2", parent=base["Heading2"], textColor=colors.HexColor(C_DARK),
                           fontSize=14, spaceBefore=14, spaceAfter=6)
    s_body = ParagraphStyle("Body", parent=base["Normal"], fontSize=10, leading=15, spaceAfter=4)
    s_bullet = ParagraphStyle("Bullet", parent=base["Normal"], fontSize=10, leading=14,
                               leftIndent=15, bulletIndent=0, spaceAfter=3)
    s_footer = ParagraphStyle("Footer", parent=base["Normal"], fontSize=8,
                               textColor=colors.grey, alignment=TA_CENTER)
    s_caption = ParagraphStyle("Caption", parent=base["Normal"], fontSize=9,
                                textColor=colors.grey, alignment=TA_CENTER, spaceAfter=6)

    story = []

    # ---- Copertina ----
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Report di Analisi Finanziaria", s_title))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(azienda, s_sub))
    if len(anni) >= 2:
        story.append(Paragraph(f"Periodo {anni[0]}–{anni[-1]}", s_sub))
    story.append(Paragraph(f"Generato il {datetime.now().strftime('%d/%m/%Y')}", s_sub))
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph(
        "Documento generato automaticamente da agente AI — "
        "verificare sempre i dati con un professionista qualificato.",
        s_footer,
    ))
    story.append(PageBreak())

    # ---- Executive Summary ----
    story.append(Paragraph("Executive Summary", s_h2))
    summary_text = commentary.get("executive_summary", "")
    summary_table = Table(
        [[Paragraph(summary_text, s_body)]],
        colWidths=[doc.width],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(C_LIGHT)),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(C_MID)),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # ---- KPI principali ----
    story.append(Paragraph("KPI Principali", s_h2))

    header_row = ["Indicatore"] + anni
    kpi_labels = [
        ("Ricavi (M€)", lambda a: f"{_v(ce.get(a, {}), 'ricavi') / 1000:.1f}"),
        ("EBITDA (M€)", lambda a: f"{_v(ce.get(a, {}), 'ebitda') / 1000:.1f}"),
        ("EBIT (M€)", lambda a: f"{_v(ce.get(a, {}), 'ebit') / 1000:.1f}"),
        ("Utile netto (M€)", lambda a: f"{_v(ce.get(a, {}), 'utile_netto') / 1000:.1f}"),
        ("EBITDA margin %", lambda a: _fmt(pa.get(a, {}).get("ebitda_margin"), "%")),
        ("EBIT margin %", lambda a: _fmt(pa.get(a, {}).get("ebit_margin"), "%")),
        ("Net margin %", lambda a: _fmt(pa.get(a, {}).get("net_margin"), "%")),
        ("PFN (M€)", lambda a: f"{_v(sp.get(a, {}), 'posizione_finanziaria_netta') / 1000:.1f}"),
        ("Patrimonio Netto (M€)", lambda a: f"{_v(sp.get(a, {}), 'patrimonio_netto') / 1000:.1f}"),
        ("PFN / EBITDA", lambda a: _fmt(pa.get(a, {}).get("pfn_ebitda"), "x")),
        ("ROI %", lambda a: _fmt(pa.get(a, {}).get("roi"), "%")),
        ("ROE %", lambda a: _fmt(pa.get(a, {}).get("roe"), "%")),
        ("Copertura interessi", lambda a: _fmt(pa.get(a, {}).get("copertura_interessi"), "x")),
    ]

    table_data = [header_row]
    for label, fn in kpi_labels:
        table_data.append([label] + [fn(a) for a in anni])
    if variazioni:
        table_data.append(["--- Variazioni YoY ---"] + [""] * len(anni))
        for var_key, var_vals in variazioni.items():
            for metric, val in var_vals.items():
                if val is not None:
                    sign = "+" if val > 0 else ""
                    table_data.append(
                        [f"  {metric.replace('_', ' ').title()} {var_key}"]
                        + [f"{sign}{val}%"] + [""] * (len(anni) - 1)
                    )

    col_widths = [doc.width * 0.45] + [doc.width * 0.55 / len(anni)] * len(anni)
    kpi_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_DARK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(C_LIGHT)]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(kpi_table)
    story.append(PageBreak())

    # ---- Grafici ----
    story.append(Paragraph("Analisi Grafica", s_h2))
    chart_files = [
        ("trend_ricavi_ebitda.png", "Trend Ricavi vs EBITDA"),
        ("marginalita.png", "Evoluzione della Marginalita'"),
        ("struttura_finanziaria.png", "Struttura Finanziaria e Leva"),
    ]
    for fname, caption in chart_files:
        fpath = str(chart_paths / fname)
        if Path(fpath).exists():
            img = Image(fpath, width=doc.width, height=doc.width * 4 / 7)
            story.append(img)
            story.append(Paragraph(caption, s_caption))
            story.append(Spacer(1, 0.3 * cm))
    story.append(PageBreak())

    # ---- Analisi ----
    story.append(Paragraph("Analisi e Raccomandazioni", s_h2))

    story.append(Paragraph("Punti di Forza", ParagraphStyle(
        "H3", parent=s_h2, fontSize=12, textColor=colors.HexColor("#1a6b1a"), spaceBefore=8)))
    for bullet in commentary.get("punti_di_forza", []):
        story.append(Paragraph(f"• {bullet}", s_bullet))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Aree di Attenzione", ParagraphStyle(
        "H3b", parent=s_h2, fontSize=12, textColor=colors.HexColor(C_RED), spaceBefore=8)))
    for bullet in commentary.get("aree_di_attenzione", []):
        story.append(Paragraph(f"• {bullet}", s_bullet))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Raccomandazioni", ParagraphStyle(
        "H3c", parent=s_h2, fontSize=12, textColor=colors.HexColor(C_DARK), spaceBefore=8)))
    for bullet in commentary.get("raccomandazioni", []):
        story.append(Paragraph(f"• {bullet}", s_bullet))

    doc.build(story)
    return outpath


@router.post("/step4")
def step4_report(session_id: str):
    _cleanup_old_sessions()
    session = _get_session(session_id)
    if not session["chart_paths"]:
        raise HTTPException(status_code=400, detail="Eseguire prima step 3.")

    client = _get_client()
    data = session["extracted_data"]
    kpis = session["kpis"]
    note = data.get("note_qualitative", "")

    try:
        commentary = _call_commentary(client, kpis, note)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Impossibile estrarre il commento AI: {exc}",
        ) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Errore AI step 4: {exc}") from exc

    report_path = _build_pdf(session_id, data, kpis, commentary)
    session["report_path"] = report_path

    anni = kpis["anni"]
    n_kpis = len(anni) * 8
    return {
        "step": 4,
        "status": "completed",
        "summary": "Report pronto",
        "details": [
            f"{4 + len(anni)} pagine circa",
            f"{len(session['chart_paths'])} grafici",
            f"{n_kpis} KPI",
        ],
        "download_url": f"/api/agent/download/{session_id}",
    }


# ---------------------------------------------------------------------------
# Task 7 — Download
# ---------------------------------------------------------------------------


@router.get("/download/{session_id}")
def download_report(session_id: str):
    _cleanup_old_sessions()
    session = _get_session(session_id)
    if not session.get("report_path") or not Path(session["report_path"]).exists():
        raise HTTPException(status_code=404, detail="Report non ancora generato.")

    azienda_slug = (
        session["extracted_data"].get("azienda", "report")
        .lower().replace(" ", "_").replace(".", "")[:30]
        if session.get("extracted_data") else "report"
    )
    filename = f"report_{azienda_slug}.pdf"

    return FileResponse(
        session["report_path"],
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
