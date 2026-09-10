# Phoenix Industries — Maintenance Copilot

Upload your machine manuals, ask questions about problems on the floor, and get
step-by-step repairs with the right diagram at the right step.

Built on the same engine as the Zynaptrix Industrial Copilot, scoped to the
knowledge side: there is no sensor monitoring, anomaly detection or model
training here.

---

## What it does

**Upload a manual.** A PDF is parsed page by page. Text becomes searchable
passages, tables are summarised, and figures are detected, split into their
individual components and described by a vision model — so a diagram can be
found by what it *shows*, not just by nearby text.

**Ask a question.** Answers are retrieved from that machine's manual and cite
what the document actually says. If a machine has no manual on file, the answer
says so rather than inventing one.

**Get guided.** "Guide me step by step" walks a technician through the repair one
step at a time, starting with safety and lockout/tagout, with the manual's
diagrams placed at the step they illustrate. Answer "done" or "I'm stuck" to move
at your own pace.

**Record what worked.** When the machine is fixed, save the fix. It is embedded
alongside the manual, so the next person asking about that machine sees the
repair that actually succeeded — not just the documentation.

---

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI (Python 3.10+) |
| UI | Next.js 16, React 19, Redux Toolkit, Tailwind 4 |
| Database | Postgres + `pgvector` (Neon) |
| Files | Cloudinary (`phoenix/` prefix) |
| Models | Qwen via Alibaba Cloud Model Studio — chat, vision, embeddings |
| Document AI | PyMuPDF, YOLOv8-DocLayNet, Mobile SAM, EasyOCR, Camelot |

---

## Setup

### 1. Database

Create a **new** Neon project — Phoenix must not share the Zynaptrix database.
Copy its connection string; `pgvector` is enabled automatically on first boot.

### 2. Backend

```powershell
cd phoenix\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env      # then fill in DATABASE_URL, AI_API_KEY, CLOUDINARY_*
python scripts
python scripts\fetch_models.py      # puts the ~90 MB layout weights in place

$env:PYTHONUTF8 = 1
python scripts\smoke_test_qwen.py   # checks all four model roles
python scripts\init_db.py           # creates the six tables
uvicorn app.main:app --port 8100
```

API docs: <http://localhost:8100/docs>

> `$env:PYTHONUTF8 = 1` is not optional on Windows. Manual content and model
> captions contain characters (`µm`, `°C`, `Ω`) that the default console encoding
> cannot represent, and a request will fail mid-flight without it. Make it
> permanent with
> `[Environment]::SetEnvironmentVariable("PYTHONUTF8","1","User")`.

Table extraction in *lattice* mode also needs [Ghostscript](https://ghostscript.com/releases/gsdnld.html)
installed. Without it, text and figures still work; ruled tables fall back to a
looser parse.

### 3. Frontend

```powershell
cd phoenix\frontend
npm install
npm run dev
```

Open <http://localhost:3100>.

Ports 8100/3100 are deliberate, so Phoenix and Zynaptrix can run at the same time.

---

## Using it

1. **Manuals** → upload a PDF with an identifier such as `PRESS_4000_V2`.
   Ingestion runs in the background; the page polls until it reports the number
   of searchable sections. A long illustrated manual can take a while, because
   every figure is analysed individually.
2. **Machines** → add a machine and link it to that manual.
3. **Ask** → select the machine and describe the problem. Use
   **Guide me step by step** for a walkthrough, and **Record fix** when it is solved.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest-manual` | Queue a manual for ingestion (returns immediately) |
| GET | `/ingest-manual/status/{manual_id}` | Ingestion progress |
| GET | `/api/manuals` | Manuals and their section counts |
| GET/POST | `/api/machines` | List / create / update machines |
| POST | `/api/machines/delete/{machine_id}` | Remove a machine |
| POST | `/api/assistant` | Ask a question (`mode`: `answer` or `wizard`) |
| GET | `/api/assistant/sessions` | Conversation history |
| GET | `/api/assistant/sessions/{id}/history` | Messages in a session |
| POST | `/api/assistant/sessions/{id}/resolve` | Record the fix that worked |
| GET | `/api/assistant/sessions/{id}/report` | Structured maintenance report |
| GET | `/health` | Liveness |

---

## Notes

**CORS.** The backend allows exactly one origin, `FRONTEND_URL`. If the UI is
served from anywhere other than that value, every request is blocked by the
browser — update `.env` when deploying.

**Provenance.** Retrieval is scoped by `manual_id`. A machine whose manual has no
ingested content produces an answer prefixed with a documentation warning, drawn
from general engineering practice. This is deliberate: answering from a different
machine's manual would be worse than admitting the gap.

**Cost.** Ingestion is the expensive step — one vision call per figure. Asking
questions is comparatively cheap. Ingest a manual once.
