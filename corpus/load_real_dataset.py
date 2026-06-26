import argparse
import json
import os


def load_hotpotqa(n: int):
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    seen = set()
    docs = []
    for row in ds:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            if title in seen:
                continue
            seen.add(title)
            text = " ".join(sentences).strip()
            if len(text) < 50:
                continue
            docs.append({"doc_id": f"doc_{len(docs)+1:03d}", "title": title, "text": text})
            if len(docs) >= n:
                return docs
    return docs


def load_nq(n: int):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/natural_questions", "default", split="validation", streaming=True)
    docs = []
    for row in ds:
        title = row["document"]["title"]
        tokens = row["document"]["tokens"]["token"]
        text = " ".join(tokens[:300]).strip()
        if len(text) < 50:
            continue
        docs.append({"doc_id": f"doc_{len(docs)+1:03d}", "title": title, "text": text})
        if len(docs) >= n:
            break
    return docs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hotpotqa", "nq"], default="hotpotqa")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    docs = load_hotpotqa(args.n) if args.source == "hotpotqa" else load_nq(args.n)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(docs)} real {args.source} documents to {out_path}")
    print("Remember to re-label eval/retrieval_eval.jsonl for this corpus.")
