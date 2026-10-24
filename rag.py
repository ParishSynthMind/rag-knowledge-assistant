"""
RAG (Retrieval-Augmented Generation) demo.

Ingests markdown documents, chunks them by section, embeds the chunks locally
(no external embeddings account needed), retrieves the most relevant chunks
for a question, and asks Claude to answer grounded only in those chunks -
with citations back to the source document, and an honest "not covered" when
the knowledge base doesn't have the answer.

Usage:
    python rag.py ingest              # build the local index from documents/
    python rag.py ask "some question" # retrieve + generate an answer
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "documents"
INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

CLAUDE_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a support assistant for Orbitly, a project-scheduling platform. \
Answer the user's question using ONLY the context chunks provided below - never use \
outside knowledge, and never guess.

Rules:
- Every factual claim must be grounded in a provided chunk. After each claim, cite its \
source in square brackets, e.g. [pricing.md].
- If the provided context does not contain enough information to answer, say so plainly: \
"I don't have that in the knowledge base" - do not fill the gap with a guess.
- Be concise and direct. No filler."""


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown doc into chunks on '## ' section headers."""
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"^#{1,2}\s+(.+)", section)
        heading = heading_match.group(1) if heading_match else source
        chunks.append({"text": section, "source": source, "heading": heading})
    return chunks


def ingest():
    model = SentenceTransformer(EMBEDDING_MODEL)
    all_chunks = []
    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        text = doc_path.read_text()
        doc_chunks = chunk_markdown(text, doc_path.name)
        all_chunks.extend(doc_chunks)
        print(f"  {doc_path.name}: {len(doc_chunks)} chunks")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2))

    print(f"\nIndexed {len(all_chunks)} chunks from {len(list(DOCS_DIR.glob('*.md')))} documents.")
    print(f"Saved to {INDEX_DIR}/")


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit("No index found - run 'python rag.py ingest' first.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = np.load(EMBEDDINGS_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text())

    query_embedding = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ query_embedding  # cosine similarity (vectors are normalized)
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for i in top_indices:
        chunk = dict(chunks[i])
        chunk["score"] = float(scores[i])
        results.append(chunk)
    return results


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']} - \"{c['heading']}\"]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> str:
    import anthropic

    # Reads ANTHROPIC_API_KEY from the environment.
    client = anthropic.Anthropic()
    context = build_context(chunks)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}",
            }
        ],
    )
    return next(b.text for b in response.content if b.type == "text")


def ask(question: str):
    chunks = retrieve(question)
    print("Retrieved chunks:")
    for c in chunks:
        print(f"  [{c['score']:.3f}] {c['source']} - {c['heading']}")
    print()

    answer = generate_answer(question, chunks)
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "ingest":
        ingest()
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python rag.py ask \"your question\"")
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)
