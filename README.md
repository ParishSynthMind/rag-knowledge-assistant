# RAG Knowledge Assistant

A document Q&A system that answers questions grounded **only** in a provided knowledge base — no hallucinated answers, and an honest "I don't know" when the knowledge base doesn't cover it.

## How it works

1. **Ingest** — markdown documents are split into chunks by section heading, then embedded locally with `sentence-transformers` (no external embeddings API or account needed).
2. **Retrieve** — a question is embedded the same way, and the most relevant chunks are pulled back by cosine similarity.
3. **Generate** — Claude answers using *only* the retrieved chunks, citing the source document after every factual claim, and explicitly says "I don't have that in the knowledge base" instead of guessing when the context doesn't cover the question.

## Why this pattern

Most "RAG demo" repos skip the part that actually matters for a production support bot: proving it won't make things up. This one is built and prompted so that every claim is traceable to a source chunk, and an out-of-scope question gets a refusal instead of a confident wrong answer — tested against both cases (see below).

## Tested behaviour

Run against a 3-document sample knowledge base for a fictional project-scheduling SaaS ("Orbitly"):
- Correctly answered a multi-step question that required combining two different sections of the docs (a refund-policy edge case).
- Correctly refused to answer a genuinely out-of-scope question rather than inventing a response.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python rag.py ingest              # build the local index from documents/
python rag.py ask "some question" # retrieve + generate an answer
```

## AI disclosure (EU AI Act Article 50)

Every answer opens with "I'm an AI assistant for Orbitly." Each `ask` run is a conversation of its own, so the line appears once per reply, before the answer. The text is set with the `AI_DISCLOSURE` environment variable (`{business}` is filled from `BUSINESS_NAME`); leaving it blank keeps the default. It is only switched off when `AI_DISCLOSURE_REMOVED_BY_WRITTEN_REQUEST` records the client's written request (date and who asked). Checked by offline tests (no model or key needed): `pip install numpy pytest && pytest`.

## Stack

Python · sentence-transformers (local embeddings) · NumPy · Anthropic Claude API

---
Part of a small portfolio of tested automation/AI builds — see [ParishSynthMind](https://github.com/ParishSynthMind) for the others.
