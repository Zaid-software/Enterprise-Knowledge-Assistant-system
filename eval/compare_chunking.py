import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingest import ingest
from eval.retrieval_eval import evaluate

CONFIGS = [
    {"chunk_size": 300, "overlap": 50},
    {"chunk_size": 800, "overlap": 100},
]


def main():
    for cfg in CONFIGS:
        print(f"\n=== chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']} ===")
        n_docs, n_chunks = ingest(**cfg)
        results = evaluate(top_k=5, role="admin")
        print(f"n_docs={n_docs}, n_chunks={n_chunks}")
        print(f"Recall@5={results['recall_at_5']}, MRR={results['mrr']}")


if __name__ == "__main__":
    main()
