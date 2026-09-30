# RAG Document Assistant — Transformer Paper QA

A retrieval-augmented generation system that answers questions grounded in
PDF documents, including tables and figures — not just plain text. Built from
scratch (no LangChain/LlamaIndex for the core RAG logic) to demonstrate a
real understanding of each pipeline stage, with a fully automated evaluation
harness to measure whether it actually works, rather than eyeballing outputs.

Test corpus: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
(Vaswani et al., 2017) — chosen deliberately for its dense tables and mixed
content types, to stress-test retrieval beyond simple prose.

## Architecture

Documents (PDF)
│
├─→ Text extraction + chunking (LangChain splitter, noise-filtered)
│
└─→ Table detection → page rendered as image → vision LLM
transcribes to clean markdown (see "Multimodal ingestion" below)
│
▼
┌──────────────┴──────────────┐
▼ ▼
Embeddings (bge-m3) BM25 keyword index
│ │
▼ │
Qdrant (vector search) │
│ │
└──────────────┬──────────────┘
▼
Hybrid Retrieval (Reciprocal Rank Fusion)
│
▼
Reranker (Jina cross-encoder)
│
▼
Grounded prompt (context + citation instructions)
│
▼
LLM generation
│
▼
Answer + citations
│
▼
Evaluation (retrieval hit rate, exact-match,
LLM-judged correctness & faithfulness, refusal handling)


Also wrapped in a FastAPI backend (`api.py`) exposing `POST /ask`.

## Stack and why

| Component | Choice | Why |
|---|---|---|
| Vector DB | Qdrant Cloud (free tier) | Purpose-built, commonly used at startups; pgvector is the other realistic choice if a team is already Postgres-first |
| Embeddings | bge-m3 (via deAPI) | Strong multilingual retrieval embedding, hosted so no local compute |
| Keyword search | BM25 (rank-bm25, local) | Catches exact terms (proper nouns, numbers) that embeddings sometimes under-match |
| Reranker | Jina cross-encoder | Cross-encoders score (query, chunk) pairs jointly — more precise than embedding similarity, run only on the small hybrid shortlist since it's slower |
| Generation LLM | NVIDIA NIM (free tier) | Free, OpenAI-compatible |
| Judge LLM | Groq (llama-3.1-8b-instant) | Deliberately a *different, faster* model than the generator — judges don't need heavy reasoning, and using the same model to grade itself is a weaker eval design |
| Vision (table transcription) | OpenRouter free-tier vision model | Multimodal ingestion for tables/figures, not just extracted text |

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

DEAPI_API_KEY=...
NVIDIA_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...


## Usage

```bash
# 1. Drop PDFs into data/
# 2. Ingest (chunks text, transcribes tables via vision LLM, embeds, stores)
python ingest.py

# 3. Ask questions via CLI
python rag.py

# 4. Or run the API
uvicorn api:app --reload
# POST http://127.0.0.1:8000/ask  {"question": "..."}

# 5. Run the evaluation suite
python evaluate.py
```

## Multimodal ingestion: why plain text extraction isn't enough

Standard PDF text extraction badly mangles tables that use "booktabs" style
(no vertical/horizontal grid lines) — common in academic papers. On this
paper's tables, plain-text extraction:
- Collapses empty cells, misaligning every value after them
- Flattens scientific notation (`3.3 · 10^18` becomes an unreadable `3.3 1018`)
- Loses row/column structure entirely on the densest table

Rather than accept that, `extract_tables.py` detects pages with a `Table N:`
caption, renders that page as an image, and asks a vision-capable LLM to
transcribe the table directly from the image into clean markdown. Figure
debris (per-word attention-visualization dumps) and the reference list are
filtered out of the text chunks entirely, since they add noise without
retrieval value.

## Evaluation: measured, not assumed

16 hand-written question/answer pairs against the paper, checked against
the actual source before being written down. Each question is scored on:
- **Retrieval hit rate** — did the right document/chunk actually get retrieved?
- **Keyword accuracy** — exact match on facts that can't be paraphrased (numbers, names)
- **LLM-judged correctness** — a second LLM checks whether the answer conveys
  the required facts regardless of wording (catches paraphrases that exact
  match would wrongly fail)
- **LLM-judged faithfulness** — does the answer only use information actually
  present in the retrieved context, with no invented claims?
- **Refusal handling** — for questions with no answer in the source, does the
  system correctly decline rather than guess?

### Before/after: does multimodal table transcription actually help?

| Metric | Text-only baseline | With vision-transcribed tables |
|---|---|---|
| Table-dependent LLM-judged correctness | **50%** (2/4) | **75%** (3/4) |
| Overall LLM-judged correctness | 85% | 92% |
| Retrieval hit rate | 100% | 100% |

On the text-only baseline, two questions produced outright generation
failures — the LLM, faced with scrambled table text, got stuck reasoning
out loud about how to parse it and never reached an answer. After adding
vision-based table transcription, both questions answered correctly and
concisely with proper citations.

The one remaining table-dependent "failure" is arguably correct behavior:
the source paper's own Table 2 has visually ambiguous column placement for
one FLOPs value, and the system declines to state which language pair it
belongs to rather than guessing. A system that faithfully refuses on
genuinely ambiguous source data is a better outcome than one that guesses
and happens to be right.

## Known limitations (found through testing, not assumed)

- **The source paper contains an internal inconsistency** (BLEU score for
  English-French listed as both 41.0 and 41.8 in different sections). The
  system correctly surfaces this conflict rather than picking one silently
  — verified via a dedicated eval question.
- **Vision-based table transcription is not deterministic.** A validation
  pass caught a run where the vision model introduced a digit transposition
  (5.29 → 5.59) and silently truncated a dense table mid-row. Fixed by
  checking `finish_reason` for truncation instead of only checking for
  empty content — a non-empty response is not the same as a complete one.
  This is documented as a real, ongoing risk: **any pipeline step involving
  a generative model, including "just" transcription, needs verification,
  not blind trust.** A more robust production version would transcribe each
  table twice and flag disagreements rather than trusting a single pass.
- **Figures without extractable text** (e.g. the attention-visualization
  diagrams) are represented only by their captions. The system can discuss
  what a figure is about but can't answer detailed questions about what a
  diagram visually shows.
- **Free-tier LLM providers are meaningfully less reliable than paid ones.**
  This project hit, in order: a mid-project model deprecation (NVIDIA
  sunset the originally-used model), embedding rate limits requiring
  backoff logic, a reasoning model silently truncating output because
  `max_tokens` didn't budget for its internal reasoning tokens (across
  three different providers/models), and a shared-pool vision model
  returning transient 429s. All are handled with retry/backoff logic in
  `ingest.py` and `extract_tables.py`, but the underlying instability is
  real and worth knowing about before relying on a free tier for anything
  time-sensitive.

## Possible extensions

- Self-consistency checking for table transcription (transcribe twice, flag disagreements)
- Hybrid weighting tuning (currently equal-weighted RRF between vector and BM25)
- A frontend (Streamlit/Next.js) instead of CLI + raw API
- Larger, multi-document corpus to test cross-document retrieval at scale