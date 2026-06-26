import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import Retriever

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET_PATH = os.path.join(BASE_DIR, "eval", "retrieval_eval_set.jsonl")


def load_eval_set(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate(top_k: int = 5, role: str = "admin"):
    retriever = Retriever()
    eval_set = load_eval_set(EVAL_SET_PATH)

    hits_at_k = 0
    reciprocal_ranks = []
    per_question_results = []

    for item in eval_set:
        question = item["question"]
        relevant = set(item["relevant_chunk_ids"])

        result = retriever.retrieve(question, top_k=top_k, user_role=role)
        retrieved_ids = [r["chunk_id"] for r in result["results"]]

        first_hit_rank = None
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant:
                first_hit_rank = rank
                break

        hit = first_hit_rank is not None
        hits_at_k += int(hit)
        reciprocal_ranks.append(1.0 / first_hit_rank if hit else 0.0)

        per_question_results.append({
            "question": question,
            "relevant_chunk_ids": list(relevant),
            "retrieved_chunk_ids": retrieved_ids,
            "hit": hit,
            "first_hit_rank": first_hit_rank,
        })

    n = len(eval_set)
    recall_at_k = hits_at_k / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0

    return {
        "n_questions": n,
        "top_k": top_k,
        f"recall_at_{top_k}": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "per_question": per_question_results,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--role", type=str, default="admin")
    ap.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "logs", "retrieval_eval_results.json"))
    args = ap.parse_args()

    results = evaluate(top_k=args.top_k, role=args.role)

    print(f"n_questions: {results['n_questions']}")
    print(f"Recall@{args.top_k}: {results[f'recall_at_{args.top_k}']}")
    print(f"MRR: {results['mrr']}")
    print()
    for pq in results["per_question"]:
        status = "HIT " if pq["hit"] else "MISS"
        print(f"[{status}] rank={pq['first_hit_rank']}  Q: {pq['question']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {args.out}")
