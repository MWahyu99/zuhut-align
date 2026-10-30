"""
zuhut_align.mlp.phase_mlp
==========================
Level 2 — PhaseMLP Adapter (Paid Tier)

Residual MLP trained with InfoNCE contrastive loss on pseudo-pairs generated
via word dropout from your own corpus documents.

Architecture:
  - Residual connection (x + net(x)) with zero-initialized last layer
  - LayerNorm + GELU + Dropout inside residual branch
  - L2-normalized output

Training:
  - Self-supervised: no labels needed
  - Input: 200–500 corpus sample documents
  - Pseudo-pairs via random 10% word dropout (N_AUGMENT=5 per doc)
  - InfoNCE contrastive loss, temperature=0.07
  - ~5 minutes on CPU regardless of corpus size (millions of docs)

This module requires: pip install torch sentence-transformers
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np

__all__ = ["PhaseMLP", "PhaseMLP_Trainer", "PhaseMLP_Adapter"]

# ── Hyperparameters ────────────────────────────────────────────────────────────
_DEFAULT_HIDDEN     = 128
_DEFAULT_DROPOUT    = 0.3
_DEFAULT_EPOCHS     = 300
_DEFAULT_LR         = 3e-4
_DEFAULT_WEIGHT_DECAY = 1e-3
_DEFAULT_TEMP       = 0.07
_DEFAULT_N_AUGMENT  = 5
_DEFAULT_DROP_RATE  = 0.10
_DEFAULT_BATCH      = 64


def _check_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PhaseMLP requires PyTorch. Install with:\n"
            "  pip install torch\n"
            "Or install the full tier:\n"
            "  pip install 'zuhut-align[mlp]'"
        )


class PhaseMLP:
    """
    Residual MLP that refines aligned embeddings for contrastive retrieval.

    Use via PhaseMLP_Trainer (to train) or PhaseMLP_Adapter (to load & infer).

    Parameters
    ----------
    dim : int
        Embedding dimension (must match your target space, default 384).
    hidden : int
        Hidden layer width. Default 128 (v3 optimal).
    dropout : float
        Dropout rate. Default 0.3 (v3 optimal).
    """

    def __init__(self, dim: int = 384, hidden: int = _DEFAULT_HIDDEN,
                 dropout: float = _DEFAULT_DROPOUT):
        torch = _check_torch()
        import torch.nn as nn
        import torch.nn.functional as F

        self._dim = dim

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, dim),
                )
                # Zero-init last layer → residual starts as identity
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def forward(self, x):
                return F.normalize(x + self.net(x), dim=-1)

        self._model = _Net()

    @property
    def model(self):
        return self._model

    def save(self, path: str | Path):
        """Save model weights to disk."""
        torch = _check_torch()
        torch.save(self._model.state_dict(), str(path))

    def load(self, path: str | Path):
        """Load model weights from disk."""
        torch = _check_torch()
        self._model.load_state_dict(
            torch.load(str(path), map_location="cpu", weights_only=True)
        )
        self._model.eval()


class PhaseMLP_Trainer:
    """
    Self-supervised trainer for PhaseMLP using corpus pseudo-pairs.

    Example
    -------
    >>> from zuhut_align import ZUHUTClient
    >>> client = ZUHUTClient(api_key="your-key")
    >>> adapter = client.create_adapter(src_dim=1024, tier="phasemlp")
    >>> adapter.train(corpus_texts, model_name="BAAI/bge-large-en-v1.5")
    >>> aligned = adapter.transform(query_vecs)
    """

    def __init__(
        self,
        linear_adapter,              # PureZeroShotAdapter (Level 1)
        hidden: int = _DEFAULT_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
        epochs: int = _DEFAULT_EPOCHS,
        lr: float = _DEFAULT_LR,
        weight_decay: float = _DEFAULT_WEIGHT_DECAY,
        temperature: float = _DEFAULT_TEMP,
        n_augment: int = _DEFAULT_N_AUGMENT,
        drop_rate: float = _DEFAULT_DROP_RATE,
        batch_size: int = _DEFAULT_BATCH,
        dim: int = 384,
        verbose: bool = True,
    ):
        self.linear_adapter = linear_adapter
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.n_augment = n_augment
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.dim = dim
        self.verbose = verbose
        self._mlp: Optional[PhaseMLP] = None

    def _augment_text(self, text: str, rng: np.random.Generator) -> str:
        """Word dropout augmentation."""
        words = text.split()
        if len(words) <= 3:
            return text
        mask = rng.random(len(words)) > self.drop_rate
        kept = [w for w, m in zip(words, mask) if m] or words
        return " ".join(kept)

    def train(self, corpus_texts: List[str], embedder) -> PhaseMLP:
        """
        Train PhaseMLP on corpus pseudo-pairs.

        Parameters
        ----------
        corpus_texts : List[str]
            200–500 representative documents from your corpus.
            Does NOT need to be the full corpus.
        embedder : callable
            Function that takes List[str] → np.ndarray of embeddings.
            E.g. sentence_transformers model.encode()

        Returns
        -------
        PhaseMLP
            Trained model (also stored as self._mlp)
        """
        torch = _check_torch()
        import torch.nn.functional as F

        rng = np.random.default_rng(42)

        # Build pseudo-pairs via word dropout augmentation
        anchors, positives = [], []
        for text in corpus_texts:
            for _ in range(self.n_augment):
                aug = self._augment_text(text, rng)
                anchors.append(text)
                positives.append(aug)

        if self.verbose:
            print(f"[PhaseMLP] Building {len(anchors)} pseudo-pairs "
                  f"from {len(corpus_texts)} docs ...")

        # Encode with source model
        anc_raw = np.array(embedder(anchors), dtype=np.float32)
        pos_raw = np.array(embedder(positives), dtype=np.float32)

        # Apply linear ZUHUT alignment first
        anc_vecs = self.linear_adapter.transform(anc_raw)
        pos_vecs = self.linear_adapter.transform(pos_raw)

        anc_t = torch.from_numpy(anc_vecs)
        pos_t = torch.from_numpy(pos_vecs)

        # Train PhaseMLP
        mlp_obj = PhaseMLP(dim=self.dim, hidden=self.hidden, dropout=self.dropout)
        net = mlp_obj.model
        net.train()

        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        N = anc_t.shape[0]
        best_loss = float("inf")

        for epoch in range(self.epochs):
            idx = torch.randperm(N)
            epoch_loss = 0.0
            steps = 0

            for i in range(0, N, self.batch_size):
                b = idx[i: i + self.batch_size]
                a = net(anc_t[b])
                p = net(pos_t[b])

                # InfoNCE loss (symmetric)
                sim = torch.matmul(a, p.T) / self.temperature
                labels = torch.arange(sim.shape[0])
                loss = (
                    F.cross_entropy(sim, labels)
                    + F.cross_entropy(sim.T, labels)
                ) / 2.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1

            avg = epoch_loss / steps
            if avg < best_loss:
                best_loss = avg

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  loss={avg:.4f}")

        net.eval()
        self._mlp = mlp_obj

        if self.verbose:
            print(f"[PhaseMLP] Training complete. Best loss: {best_loss:.4f}")

        return mlp_obj

    @property
    def mlp(self) -> Optional[PhaseMLP]:
        return self._mlp


class PhaseMLP_Adapter:
    """
    Inference adapter: Linear ZUHUT → PhaseMLP pipeline.

    Parameters
    ----------
    linear_adapter : PureZeroShotAdapter
        Pre-built Level 1 adapter.
    mlp : PhaseMLP
        Trained PhaseMLP model.
    """

    def __init__(self, linear_adapter, mlp: PhaseMLP):
        self.linear_adapter = linear_adapter
        self.mlp = mlp
        self.mlp.model.eval()

    def transform(self, vecs: np.ndarray) -> np.ndarray:
        """
        Full pipeline: source embeddings → ZUHUT aligned → PhaseMLP refined.

        Parameters
        ----------
        vecs : np.ndarray, shape (N, src_dim)

        Returns
        -------
        np.ndarray, shape (N, 384)
            L2-normalized embeddings ready for cosine similarity search.
        """
        torch = _check_torch()
        import torch.nn.functional as F

        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]

        # Step 1: Linear ZUHUT
        aligned = self.linear_adapter.transform(vecs)

        # Step 2: PhaseMLP refinement
        with torch.no_grad():
            t = torch.from_numpy(aligned)
            out = self.mlp.model(t)
            return out.numpy()

    def save(self, path: str | Path):
        """Save PhaseMLP weights to disk."""
        self.mlp.save(path)

    @classmethod
    def load(cls, path: str | Path, linear_adapter, dim: int = 384,
             hidden: int = _DEFAULT_HIDDEN, dropout: float = _DEFAULT_DROPOUT):
        """Load a previously saved PhaseMLP_Adapter from disk."""
        mlp = PhaseMLP(dim=dim, hidden=hidden, dropout=dropout)
        mlp.load(path)
        return cls(linear_adapter, mlp)
