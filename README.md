# 🧠 Simple RAG Pipeline — From Scratch

A clean, beginner-friendly Retrieval-Augmented Generation (RAG) pipeline built from scratch on Google Colab with a free T4 GPU.

```
PDF
 ↓  PyMuPDF          — extract text page by page
 ↓  LangChain        — split into overlapping chunks
 ↓  MiniLM (BERT)    — embed chunks into 384-dim vectors
 ↓  FAISS            — store & search vectors by cosine similarity
 ↓  Mistral-7B       — generate answer from retrieved context

```

---

## 📁 Project Structure

```
rag_simple_fixed.ipynb   — main notebook (11 cells)
README.md                — this file
```

---

## ⚙️ Stack

| Component | Tool | Why |
|---|---|---|
| PDF Extraction | PyMuPDF (`fitz`) | Fast, accurate text extraction |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Preserves paragraph structure |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim BERT-style, fast & accurate |
| Vector DB | FAISS `IndexFlatIP` | Exact cosine search, perfect for <100K chunks |
| Generator | `mistralai/Mistral-7B-Instruct-v0.2` (4-bit) | Best open 7B model, fits T4 VRAM |
| Framework | LangChain (text splitting + document schema) | Clean abstractions |

---



### 1. Open in Google Colab
Upload `rag_simple_fixed.ipynb` to [colab.research.google.com](https://colab.research.google.com)

### 2. Switch to T4 GPU
```
Runtime → Change runtime type → T4 GPU
```

### 3. Upload your PDF
```
Click 📁 folder icon (left sidebar) →  upload → select your PDF
```

### 4. Run cells in order
```
Cell 1  — Install packages
Cell 2  — All imports + GPU check
Cell 3  — Config (hyperparameters)
Cell 4  — Upload & extract PDF
Cell 5  — Chunk text
Cell 6  — Embed chunks
Cell 7  — Build FAISS index + test retrieval
Cell 8  — Load Mistral-7B (3–5 min)
Cell 9  — Define ask() function
Cell 10 — Ask a single question
Cell 11 — Interactive Q&A loop
```

---

##  Hyperparameters

| Parameter | Value | What it controls |
|---|---|---|
| `CHUNK_SIZE` | 400 | Characters per chunk. Too small = no context. Too large = noise |
| `CHUNK_OVERLAP` | 50 | Shared chars between chunks. Prevents info loss at boundaries |
| `EMBED_DIM` | 384 | Vector size output by MiniLM |
| `TOP_K` | 3 | Number of chunks retrieved per query |
| `SCORE_THRESH` | 0.15 | Min cosine similarity to include a chunk (lower for technical PDFs) |
| `MAX_NEW_TOKENS` | 512 | Max length of generated answer |
| `TEMPERATURE` | 0.3 | Lower = more factual, higher = more creative |

### Tuning `SCORE_THRESH` by document type

| PDF Type | Recommended Threshold |
|---|---|
| General text, articles | 0.25 – 0.35 |
| Research papers, technical docs | 0.10 – 0.20 |
| Legal / medical documents | 0.15 – 0.25 |

---

## 🔍 How Retrieval Works

```python
# 1. Embed the question into a 384-dim vector
q_vec = embed_model.encode(["What is attention?"], normalize_embeddings=True)

# 2. Search FAISS — returns top K chunks by cosine similarity
scores, indices = index.search(q_vec, top_k=3)

# 3. Filter by score threshold
# scores: [0.28, 0.21, 0.19, 0.07]  →  keep only >= 0.15
# result: 3 relevant chunks passed to Mistral
```

**Why cosine similarity and not softmax?**
Embeddings are L2-normalised before storing in FAISS. This means dot product = cosine similarity directly. Scores land in `[-1, 1]` — no softmax needed. Softmax is only for raw logits (classification heads), not similarity search.

---

## 🤖 How Generation Works

Mistral uses the `[INST]` chat format:

```
[INST] You are a helpful assistant. Answer using ONLY the context below.

Context:
[Page 4 | Score 0.28]
The Transformer model relies entirely on self-attention...

[Page 2 | Score 0.26]
We propose a new architecture called the Transformer...

Question: What is the main contribution of this paper? [/INST]
```

Mistral then generates the answer **grounded in your PDF**, not from its training memory.

---
---

## 💡 Tips

- **After any kernel reset** always re-run Cells 1 → 2 → 3 before anything else
- **Cell 8 takes 3–5 minutes** — Mistral-7B is downloading ~4GB (4-bit quantized)
- **Scanned PDFs won't work** — PyMuPDF reads text layers, not images. Use text-based PDFs
- **Password-protected PDFs won't open** — remove protection first
- **Best PDF size** — 5 to 100 pages works smoothly on free Colab T4

---

## 🗺️ What's Next — Adding LangGraph

Once the simple pipeline works, LangGraph adds a stateful graph on top:

```
START
  ↓
[retrieve]   — embed query → FAISS → top K chunks
  ↓
[grade]      — are chunks relevant? yes/no
  ↓
[generate]   — build prompt → Mistral → answer
  OR
[fallback]   — honest "not found" response
  ↓
END
```

Benefits over the simple pipeline:
- Automatic fallback when no relevant chunks found
- Easy to add query rewriting, re-ranking, or multi-hop retrieval
- Each step is modular and independently testable

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
