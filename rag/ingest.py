import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunking import chunk_document, extract_min_role, strip_role_marker
from rag.embeddings import get_embedder
from rag.vector_store import get_vector_store

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, "corpus", "documents.jsonl")
STORE_DIR = os.path.join(BASE_DIR, "rag", "store")


def load_corpus(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def ingest(chunk_size: int = 300, overlap: int = 50):
    os.makedirs(STORE_DIR, exist_ok=True)
    docs = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(docs)} documents from corpus.")

    all_chunks = []
    for d in docs:
        min_role = extract_min_role(d["text"])
        clean_text = strip_role_marker(d["text"])
        chunks = chunk_document(d["doc_id"], d["title"], clean_text,
                                 chunk_size=chunk_size, overlap=overlap, min_role=min_role)
        all_chunks.extend(chunks)
    print(f"Produced {len(all_chunks)} chunks (chunk_size={chunk_size}, overlap={overlap}).")

    embedder = get_embedder()
    print(f"Using embedder: {embedder.name} (dim={embedder.dim})")

    texts = [c.text for c in all_chunks]
    vectors = embedder.embed(texts)

    store = get_vector_store(embedder.dim)
    print(f"Using vector store backend: {store.backend}")
    store.add(vectors, [c.chunk_id for c in all_chunks])
    store.save(os.path.join(STORE_DIR, "vector_index"))

    # persist chunk metadata (id -> title/text/doc_id/min_role) for retrieval-time lookup
    meta = {
        c.chunk_id: {"doc_id": c.doc_id, "title": c.title, "text": c.text, "min_role": c.min_role}
        for c in all_chunks
    }
    with open(os.path.join(STORE_DIR, "chunk_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(STORE_DIR, "ingest_config.json"), "w") as f:
        json.dump({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedder": embedder.name,
            "vector_store_backend": store.backend,
            "n_docs": len(docs),
            "n_chunks": len(all_chunks),
        }, f, indent=2)

    print(f"Ingestion complete. Store written to {STORE_DIR}")
    return len(docs), len(all_chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)
