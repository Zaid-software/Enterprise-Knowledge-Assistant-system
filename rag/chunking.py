import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    min_role: str = "intern"


def _words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def chunk_document(doc_id: str, title: str, text: str, chunk_size: int = 300, overlap: int = 50,
                    min_role: str = "intern") -> List[Chunk]:
    words = _words(text)
    if len(words) <= chunk_size:
        return [Chunk(chunk_id=f"{doc_id}::c0", doc_id=doc_id, title=title, text=text, min_role=min_role)]

    chunks = []
    start = 0
    idx = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        window = words[start:start + chunk_size]
        chunk_text = " ".join(window)
        chunks.append(Chunk(chunk_id=f"{doc_id}::c{idx}", doc_id=doc_id, title=title, text=chunk_text, min_role=min_role))
        idx += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


_INTERNAL_ROLE_MARKER = re.compile(r"min_role:\s*(intern|manager|admin)", re.IGNORECASE)


def extract_min_role(text: str) -> str:
    m = _INTERNAL_ROLE_MARKER.search(text)
    return m.group(1).lower() if m else "intern"


def strip_role_marker(text: str) -> str:
    return _INTERNAL_ROLE_MARKER.sub("", text).strip()
