# RAG From Scratch

A minimal, working Retrieval-Augmented Generation pipeline you can read top to bottom in one sitting. It indexes documents, does semantic search, retrieves relevant context, and generates a grounded answer.

## Quickstart

```bash
pip install -r requirements.txt
python3 rag_system.py
```

This indexes the three sample docs in `sample_docs/`, runs three test queries, and prints the retrieved chunks plus the assembled prompt. Without an API key it uses `EchoLLM`, a stub that shows you exactly what would be sent to a real model so you can verify the whole pipeline works before spending any API credits.

To get real generated answers:

```bash
export ANTHROPIC_API_KEY=sk-...
pip install anthropic
python3 rag_system.py
```

## How it works

```
documents  →  chunk  →  embed  →  index (FAISS)
                                        │
query  →  embed  →  search top-k  ──────┘
                        │
              context + query → prompt → LLM → grounded answer
```

**1. Chunking** (`chunk_text`) — splits documents into overlapping word windows. Overlap keeps a sentence from being severed across a chunk boundary and becoming unretrievable.

**2. Embedding** (`EmbeddingModel`) — two backends, swappable:
- `TfidfEmbedder`: pure numpy/sklearn, no downloads, runs anywhere instantly. This is the default so the demo works offline out of the box.
- `SentenceTransformerEmbedder`: real dense semantic embeddings (`all-MiniLM-L6-v2`). Swap this in for genuine semantic matching (e.g. it can match "car" to "automobile", which TF-IDF can't).

**3. Vector store** (`VectorStore`) — a FAISS `IndexFlatIP` (cosine similarity via inner product on normalized vectors). Simple, exact, fast enough for up to ~1M vectors on one machine.

**4. Retrieval** (`RAGPipeline.retrieve`) — embeds the query and pulls the top-k nearest chunks.

**5. Generation** (`RAGPipeline.answer`) — stuffs retrieved chunks into a prompt template that instructs the model to answer *only* from context (this is what reduces hallucination), and calls the LLM.

## Swapping in production-grade pieces

| Component | This project | Swap for |
|---|---|---|
| Embeddings | TF-IDF (offline demo) | `sentence-transformers`, OpenAI `text-embedding-3`, Voyage AI |
| Vector store | FAISS (local, in-memory) | ChromaDB (persistence + metadata filtering), Pinecone, Weaviate |
| LLM | EchoLLM stub / Claude | Any Anthropic/OpenAI model, or a local model via Ollama |
| Orchestration | Plain Python classes | LangChain `RetrievalQA` / LCEL chains, LlamaIndex |

Each piece is an isolated class (`EmbeddingModel`, `VectorStore`, `LLM`) specifically so you can swap one without touching the others.

## Files

- `rag_system.py` — the full pipeline (chunking, embedding, FAISS store, retrieval, generation)
- `sample_docs/` — three short .txt files used by the demo
- `requirements.txt` — core deps installed, optional deps commented out

## Extending this

- **Add your own docs**: drop `.txt` files into `sample_docs/`, or point `load_documents()` at any folder.
- **Better chunking**: swap the word-count splitter for a recursive/semantic splitter (e.g. LangChain's `RecursiveCharacterTextSplitter`) if your docs have structure (headings, code blocks).
- **Hybrid search**: combine the TF-IDF sparse score with dense embedding similarity for better recall on rare keywords (product codes, names) that dense embeddings sometimes miss.
- **Re-ranking**: add a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) after retrieval to reorder the top-k by relevance before generation.
- **Citations**: the prompt already numbers sources (`[1]`, `[2]`, ...) — ask the LLM to cite them inline for traceable answers.

  <img width="1917" height="1018" alt="Screenshot 2026-08-27 111415" src="https://github.com/user-attachments/assets/38c4cd18-c204-4d82-adf4-c5bc2c2dce2b" />

