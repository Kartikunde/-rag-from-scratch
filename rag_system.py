"""
RAG From Scratch
================
A minimal, dependency-light Retrieval-Augmented Generation pipeline.

Pipeline stages:
  1. Load & chunk documents
  2. Embed chunks              -> EmbeddingModel
  3. Index embeddings          -> VectorStore (FAISS)
  4. Retrieve top-k on query   -> semantic search
  5. Generate grounded answer  -> LLM + retrieved context

Two embedding backends are included:
  - TfidfEmbedder:        pure numpy/sklearn, no downloads, runs anywhere. Good default
                           for a fast local demo or offline/air-gapped use.
  - SentenceTransformerEmbedder: real dense semantic embeddings (all-MiniLM-L6-v2).
                           Swap in when you want stronger semantic matching and have
                           `sentence-transformers` installed.

Two LLM backends are included:
  - EchoLLM:      no API key needed, just shows you the assembled prompt + a stub
                   answer, so you can test the pipeline end-to-end offline.
  - AnthropicLLM: calls the real Claude API for generation, given ANTHROPIC_API_KEY.

Everything is written as small, swappable classes so you can drop in your own
embedding model, vector store (Chroma, Pinecone, Weaviate, ...), or LLM without
touching the rest of the pipeline.
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# 1. Document loading & chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: int
    text: str
    source: str


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    """Split text into overlapping word-based chunks.

    Overlap matters: it stops an answer-bearing sentence from being severed
    across a chunk boundary and lost to retrieval.
    """
    words = re.split(r"\s+", text.strip())
    chunks = []
    start = 0
    cid = 0
    while start < len(words):
        end = start + chunk_size
        piece = " ".join(words[start:end])
        if piece:
            chunks.append(Chunk(id=cid, text=piece, source=source))
            cid += 1
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def load_documents(paths: Sequence[str]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    next_id = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for c in chunk_text(text, source=os.path.basename(path)):
            c.id = next_id
            next_id += 1
            all_chunks.append(c)
    return all_chunks


# ---------------------------------------------------------------------------
# 2. Embedding models (swappable)
# ---------------------------------------------------------------------------

class EmbeddingModel:
    dim: int

    def fit(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(EmbeddingModel):
    """Lightweight, offline embedder. Not 'real' semantic embeddings, but a
    fast, dependency-light stand-in that demonstrates the full pipeline
    without downloading a model. Swap for SentenceTransformerEmbedder for
    genuine semantic search quality."""

    def __init__(self, max_features: int = 4096):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.dim = max_features

    def fit(self, texts: Sequence[str]) -> None:
        self.vectorizer.fit(texts)
        self.dim = len(self.vectorizer.vocabulary_)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEmbedder(EmbeddingModel):
    """Real dense semantic embeddings. Requires: pip install sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def fit(self, texts: Sequence[str]) -> None:
        pass  # pretrained, nothing to fit

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


# ---------------------------------------------------------------------------
# 3. Vector store (FAISS)
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, dim: int):
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> List[tuple[Chunk, float]]:
        scores, idxs = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


# ---------------------------------------------------------------------------
# 4. LLM backends (swappable)
# ---------------------------------------------------------------------------

class LLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class EchoLLM(LLM):
    """No API key required. Lets you verify retrieval + prompt assembly
    end-to-end before wiring up a real model."""

    def generate(self, prompt: str) -> str:
        return (
            "[EchoLLM stub - no API call made]\n"
            "This is what would be sent to the LLM:\n\n" + prompt
        )


class AnthropicLLM(LLM):
    """Calls the real Claude API. Requires ANTHROPIC_API_KEY in the environment
    and `pip install anthropic`."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic  # lazy import
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


# ---------------------------------------------------------------------------
# 5. The RAG pipeline itself
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say you don't know instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGPipeline:
    embedder: EmbeddingModel
    llm: LLM
    store: VectorStore = field(init=False, default=None)
    top_k: int = 4

    def index(self, chunks: List[Chunk]) -> None:
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        self.store = VectorStore(dim=vectors.shape[1])
        self.store.add(chunks, vectors)

    def retrieve(self, query: str, k: int | None = None) -> List[tuple[Chunk, float]]:
        if self.store is None:
            raise RuntimeError("Call .index() before retrieving.")
        qvec = self.embedder.encode([query])
        return self.store.search(qvec, k=k or self.top_k)

    def answer(self, query: str, k: int | None = None) -> dict:
        results = self.retrieve(query, k=k)
        context = "\n\n".join(
            f"[{i+1}] (source: {c.source})\n{c.text}" for i, (c, _score) in enumerate(results)
        )
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
        answer_text = self.llm.generate(prompt)
        return {
            "query": query,
            "retrieved": results,
            "prompt": prompt,
            "answer": answer_text,
        }


# ---------------------------------------------------------------------------
# Demo / CLI usage
# ---------------------------------------------------------------------------

def _demo():
    docs_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    paths = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.endswith(".txt")]

    print(f"Loading {len(paths)} documents...")
    chunks = load_documents(paths)
    print(f"Created {len(chunks)} chunks.")

    use_real_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    embedder = TfidfEmbedder()
    llm = AnthropicLLM() if use_real_llm else EchoLLM()

    pipeline = RAGPipeline(embedder=embedder, llm=llm, top_k=3)
    pipeline.index(chunks)

    queries = [
        "What is retrieval-augmented generation?",
        "Why does chunking use overlap?",
        "What vector store does this project use?",
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print("QUERY:", q)
        result = pipeline.answer(q)
        print("\nRetrieved chunks:")
        for chunk, score in result["retrieved"]:
            preview = textwrap.shorten(chunk.text, width=100)
            print(f"  score={score:.3f} source={chunk.source} -> {preview}")
        print("\nAnswer:")
        print(result["answer"] if use_real_llm else "(EchoLLM stub — set ANTHROPIC_API_KEY for a real generated answer)")


if __name__ == "__main__":
    _demo()
