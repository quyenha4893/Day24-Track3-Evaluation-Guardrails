import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ─── CONFIG SAFE IMPORT ─────────────────────────────────

try:
    from config import (
        SEMANTIC_THRESHOLD,
        HIERARCHICAL_PARENT_SIZE,
        HIERARCHICAL_CHILD_SIZE,
        EMBEDDING_MODEL,
    )
except ImportError:
    SEMANTIC_THRESHOLD = 0.75
    HIERARCHICAL_PARENT_SIZE = 2000
    HIERARCHICAL_CHILD_SIZE = 500
    EMBEDDING_MODEL = "BAAI/bge-m3"


# ─── CORE DATA STRUCTURE ────────────────────────────────

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


# ─── BASELINE CHUNKING ──────────────────────────────────

def chunk_basic(text: str, chunk_size: int = 500, metadata=None) -> list[Chunk]:
    metadata = metadata or {}
    paragraphs = [p.strip() for p in re.split(r'\n\n|\n', text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"

    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))

    return chunks


# ─── MODEL CACHE ────────────────────────────────────────

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


# ─── Strategy 1: Semantic ───────────────────────────────

def chunk_semantic(text: str, threshold: float = None,
                   metadata: dict | None = None) -> list[Chunk]:

    metadata = metadata or {}
    threshold = threshold or SEMANTIC_THRESHOLD

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    model = get_model()
    embeddings = model.encode(sentences)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])

    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))

    return chunks


# ─── Strategy 2: Hierarchical ───────────────────────────

def chunk_hierarchical(text: str,
                       parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None):

    metadata = metadata or {}

    paragraphs = [p.strip() for p in re.split(r'\n\n|\n', text) if p.strip()]
    parents, children = [], []

    current = ""
    p_index = 0

    for para in paragraphs:
        if len(current) + len(para) > parent_size and current:
            pid = f"parent_{p_index}"
            parents.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current = ""
            p_index += 1
        current += para + "\n\n"

    if current.strip():
        pid = f"parent_{p_index}"
        parents.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    for parent in parents:
        pid = parent.metadata["parent_id"]
        text = parent.text

        for i in range(0, len(text), child_size):
            child_text = text[i:i + child_size]
            if child_text.strip():
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid
                ))

    return parents, children


# ─── Strategy 3: Structure-Aware ────────────────────────

def chunk_structure_aware(text: str, metadata: dict | None = None):

    metadata = metadata or {}

    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_content.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))

    return chunks


# ─── Compare ────────────────────────────────────────────

def _compute_stats(chunks: list[Chunk]):
    if not chunks:
        return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": int(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }

# ─── DOCUMENT LOADER ────────────────────────────────────


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(paths: list[str] | None = None) -> list[dict]:
    """
    Load documents from file paths or auto-scan ./data/ folder.
    Supports: .pdf, .txt, .md
    """
    if not paths:
        # Data lives in rag/data/ (sibling of this file)
        data_dir = Path(__file__).parent / "data"
        if not data_dir.exists():
            # Fallback: project-root/data
            data_dir = Path(__file__).parent.parent / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"data/ folder not found at: {data_dir}")
        paths = [
            str(p) for p in sorted(data_dir.iterdir())
            if p.suffix.lower() in {".pdf", ".txt", ".md"}
        ]
        if not paths:
            raise FileNotFoundError(f"No .pdf/.txt/.md files found in: {data_dir}")

    documents = []
    for path in paths:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".pdf":
                text = _read_pdf(path)
            elif ext in {".txt", ".md"}:
                text = _read_txt(path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": path,
                        "filename": Path(path).name
                    }
                })
        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

    return documents

def compare_strategies(documents: list[dict]):

    results = {
        "basic": [],
        "semantic": [],
        "hierarchical_parents": [],
        "hierarchical_children": [],
        "structure": []
    }

    for doc in documents:
        text = doc["text"]
        metadata = doc.get("metadata", {})

        results["basic"].extend(chunk_basic(text, metadata=metadata))
        results["semantic"].extend(chunk_semantic(text, metadata=metadata))

        parents, children = chunk_hierarchical(text, metadata=metadata)
        results["hierarchical_parents"].extend(parents)
        results["hierarchical_children"].extend(children)

        results["structure"].extend(chunk_structure_aware(text, metadata=metadata))

    return {
        "basic": _compute_stats(results["basic"]),
        "semantic": _compute_stats(results["semantic"]),
        "hierarchical": {
            "parents": _compute_stats(results["hierarchical_parents"]),
            "children": _compute_stats(results["hierarchical_children"])
        },
        "structure": _compute_stats(results["structure"])
    }
