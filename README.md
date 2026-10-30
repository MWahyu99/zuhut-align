# zuhut-align

> **Cross-model embedding alignment.** Bridge any two embedding models — zero-shot, no labels, no reindexing.

[![PyPI](https://img.shields.io/badge/pypi-zuhut--align-blue)](https://pypi.org/project/zuhut-align)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## The Problem

You have **millions of documents** indexed with `bge-large-en-v1.5` (1024d).  
You want to query with `MiniLM` (384d) — faster, cheaper, fits on edge.

Without alignment: **R@5 = 0.13** (unusable).  
With ZUHUT: **R@5 = 0.70** (91% of same-model ceiling) — zero reindexing.

---

## Install

```bash
# Level 1 — ZeroShot (free, no GPU needed)
pip install zuhut-align

# Level 2 — PhaseMLP training (requires PyTorch)
pip install 'zuhut-align[mlp]'

# Full install
pip install 'zuhut-align[full]'
```

---

## Quick Start

### Level 1 — ZeroShot (Free)

```python
from zuhut_align import ZUHUTClient

# No API key needed
client = ZUHUTClient()
adapter = client.create_adapter(src_dim=1024)   # bge-large → MiniLM space

# Align your corpus vectors once (store the result in Qdrant/FAISS/etc.)
corpus_aligned = adapter.transform(corpus_vectors_1024d)   # shape (N, 384)

# Align query at search time
query_aligned = adapter.transform(query_vector_1024d)      # shape (1, 384)

# Now search corpus_aligned with query_aligned using cosine similarity
```

**Result:** R@5 jumps from 0.13 → 0.40. Zero training. Zero data. Instant.

---

### Level 2 — PhaseMLP (Paid, ~5 min training)

```python
from zuhut_align import ZUHUTClient
from sentence_transformers import SentenceTransformer

model_b = SentenceTransformer("paraphrase-MiniLM-L6-v2")

client  = ZUHUTClient(api_key="zuhut-your-key-here")
trainer = client.create_adapter(src_dim=1024, tier="phasemlp")

# Train on 200–500 sample docs from your corpus (NOT the full corpus)
# Training time: ~5 min on CPU regardless of corpus size
trainer.train(
    corpus_texts=sample_docs,     # List[str], 200–500 docs
    embedder=model_b.encode,      # Your query-side encoder
)

# Now trainer is a PhaseMLP_Trainer with a trained .mlp
adapter = trainer   # .transform() is available after training

corpus_aligned = adapter.transform(corpus_vectors_1024d)
query_aligned  = adapter.transform(query_vector_1024d)
```

**Result:** R@5 = **0.70** — 91% of same-model ceiling.

---

### Full Qdrant Example

```python
from zuhut_align import ZUHUTClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Models
model_a = SentenceTransformer("BAAI/bge-large-en-v1.5")    # corpus model
model_b = SentenceTransformer("paraphrase-MiniLM-L6-v2")   # query model

# 2. ZUHUT alignment (zero-shot)
client  = ZUHUTClient()
adapter = client.create_adapter(src_dim=1024)

# 3. Encode & align corpus
corpus_vecs   = model_a.encode(corpus_docs, normalize_embeddings=True)
corpus_zuhut  = adapter.transform(corpus_vecs)   # shape (N, 384)

# 4. Index into Qdrant (384d, one-time)
qdrant = QdrantClient(host="localhost", port=6333)
qdrant.recreate_collection("my_corpus", vectors_config=VectorParams(
    size=384, distance=Distance.COSINE
))
qdrant.upsert("my_corpus", points=[
    PointStruct(id=i, vector=corpus_zuhut[i].tolist())
    for i in range(len(corpus_docs))
])

# 5. Search with cheap query model
query    = "how do neural networks learn?"
q_vec    = model_b.encode(query, normalize_embeddings=True)
q_zuhut  = adapter.transform(q_vec)

results  = qdrant.query_points("my_corpus", query=q_zuhut[0].tolist(), limit=5)
```

## Use Cases

adapter.transform(vecs) returns plain numpy — use it anywhere:

1. Cross-model search without reindexing Corpus indexed with BGE/OpenAI/Cohere → query with a different model → no need to re-encode millions of documents.

2. Switch embedding models without downtime Migrate from old model to new model gradually — old corpus still works, new query model plugs in via ZUHUT.

3. Cut API embedding costs Index corpus once with expensive model (OpenAI ada-002) → query forever with a free local model (MiniLM).

4. Multi-model search One corpus, queryable from multiple different models simultaneously — mobile uses small model, server uses large model.

5. Edge / embedded device search Corpus on server with large model → lightweight device (Raspberry Pi, mobile) queries with small model → ZUHUT bridges the gap.

6. Works with any vector database Output is plain numpy → drop into Qdrant, FAISS, ChromaDB, LanceDB, Weaviate, Milvus, or any cosine similarity search.

7. Multilingual cross-model English corpus (EN model) → query in Indonesian or other languages (multilingual model) → still retrieves relevant results.

8. Model evaluation & benchmarking Compare retrieval results across different models in a unified space without dimensional bias.

---

## How It Works

```
Source Model (1024d)
       ↓
   SVD Projection → 384d
       ↓
 Procrustes Rotation  ← synthetic anchors (N=250, SEED=42)
       ↓
  L2 Normalize
       ↓
  [PhaseMLP] ← optional, trained on your corpus pseudo-pairs
       ↓
Target Space (384d) — compatible with your indexed corpus
```

### Zero-Shot (Level 1)

- Generates 250 synthetic Gaussian anchor pairs
- Learns an SVD projection (dim reduction) + Procrustes rotation
- No real data, no training, reproducible across runs

### PhaseMLP (Level 2)

- Residual MLP (x + net(x)) with zero-init last layer
- Trained with InfoNCE contrastive loss on corpus pseudo-pairs
- Pseudo-pairs via word dropout (10% random word removal)
- Training invariant to corpus size: 200–500 docs → 5 min on CPU

---

## Benchmark Results

Tested on Qdrant with BGE-large (corpus) vs MiniLM (query):

| Condition                         | R@5       | vs Ceiling |
| --------------------------------- | --------- | ---------- |
| Same-model baseline (ceiling)     | 0.767     | —          |
| Cross-model, no alignment         | 0.133     | -63.3%     |
| **+ ZUHUT ZeroShot (Level 1)**    | **0.400** | **-36.7%** |
| **+ ZUHUT PhaseMLP v3 (Level 2)** | **0.700** | **-6.7%**  |

Vector databases tested: Qdrant ✓, FAISS ✓, LanceDB ✓, ChromaDB ✓

---

## Tiers

|                     | Level 1 ZeroShot | Level 2 PhaseMLP | Level 3 Engine |
| ------------------- | ---------------- | ---------------- | -------------- |
| **Price**           | Free             | Paid             | Enterprise     |
| **Training data**   | None             | 200–500 docs     | None           |
| **Training time**   | 0s               | ~5 min CPU       | 0s             |
| **R@5 gain**        | +30%             | +57%             | +57%           |
| **Inference speed** | Fast (NumPy)     | Fast (PyTorch)   | Ultra (Rust)   |
| **API key**         | Not required     | Required         | License key    |

---

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- scipy ≥ 1.10
- torch ≥ 2.0 _(Level 2 only)_
- sentence-transformers ≥ 2.7 _(Level 2 only)_

---

## License

Apache 2.0 — Python SDK is open source.  
The Rust engine (`zuhri_engine`) is proprietary (Level 3 / Enterprise).

---

## Links

- 🌐 Website: [zuhut.ai](https://zuhut.ai) _(coming soon)_
- 📖 Docs: [docs.zuhut.ai](https://docs.zuhut.ai) _(coming soon)_
- 📧 Enterprise: alam09yahoo.com@gmail.com
- 🐛 Issues: [github.com/MWahyu99/zuhut-align/issues](https://github.com/MWahyu99/zuhut-align/issues)
