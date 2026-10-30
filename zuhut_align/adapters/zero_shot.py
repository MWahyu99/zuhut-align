"""
zuhut_align.adapters.zero_shot
==============================
Level 1 — PureZeroShotAdapter (Free Tier)

SVD projection + Centered Procrustes rotation using synthetic Gaussian anchors.
No training data required. Works out-of-the-box across any two embedding models.

Algorithm:
  1. Generate N synthetic anchor pairs (same SEED → reproducible)
  2. Center both anchor clouds
  3. If src_dim ≠ tgt_dim: reduce src via SVD truncation (W_svd)
  4. Estimate optimal rotation via Procrustes (SVD of cross-covariance M)
  5. Apply: center → project → normalize → rotate → normalize
"""

import numpy as np
from scipy.linalg import svd as scipy_svd

from ..utils.normalize import l2_normalize

__all__ = ["PureZeroShotAdapter"]

# ── Defaults (match training experiments) ─────────────────────────────────────
_N_ANCHORS  = 250
_TARGET_DIM = 384
_SEED       = 42


class PureZeroShotAdapter:
    """
    Zero-shot cross-model embedding adapter.

    Bridges any source embedding model to a 384-dim target space using
    only synthetic Gaussian anchors — no real training data needed.

    Parameters
    ----------
    src_dim : int
        Dimensionality of the source model (e.g. 1024 for bge-large).
    tgt_dim : int
        Target dimensionality. Default 384 (matches MiniLM / ZUHUT space).
    n_anchors : int
        Number of synthetic anchor pairs. Default 250.
    seed : int
        Random seed for reproducibility. Default 42.

    Example
    -------
    >>> from zuhut_align import ZUHUTClient
    >>> client = ZUHUTClient()
    >>> adapter = client.create_adapter(src_dim=1024, tier="zeroshot")
    >>> aligned = adapter.transform(corpus_vectors)   # shape (N, 384)
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int = _TARGET_DIM,
        n_anchors: int = _N_ANCHORS,
        seed: int = _SEED,
    ):
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim

        rng = np.random.default_rng(seed)
        src_anchors = l2_normalize(
            rng.standard_normal((n_anchors, src_dim)).astype(np.float32)
        )
        tgt_anchors = l2_normalize(
            rng.standard_normal((n_anchors, tgt_dim)).astype(np.float32)
        )

        # Store means for centering at transform time
        self._src_mean = src_anchors.mean(axis=0, keepdims=True)
        self._tgt_mean = tgt_anchors.mean(axis=0, keepdims=True)

        src_c = src_anchors - self._src_mean
        tgt_c = tgt_anchors - self._tgt_mean

        # Step 1: dimensionality reduction via SVD (only if dims differ)
        if src_dim != tgt_dim:
            U, S, Vt = scipy_svd(src_c, full_matrices=False)
            k = min(tgt_dim, Vt.shape[0])
            W = Vt[:k, :].T  # shape (src_dim, k)
            if k < tgt_dim:
                pad = np.zeros((W.shape[0], tgt_dim - k), dtype=np.float32)
                W = np.hstack([W, pad])
            self._W_svd = W.astype(np.float32)
            proj_anchors = l2_normalize(src_c @ self._W_svd)
        else:
            self._W_svd = None
            proj_anchors = l2_normalize(src_c)

        # Step 2: Procrustes rotation
        M = proj_anchors.T @ tgt_c
        U_p, _, Vt_p = scipy_svd(M, full_matrices=False)
        self._R = (Vt_p.T @ U_p.T).astype(np.float32)

    def transform(self, vecs: np.ndarray) -> np.ndarray:
        """
        Align source embeddings to the ZUHUT target space.

        Parameters
        ----------
        vecs : np.ndarray, shape (N, src_dim)
            Source model embeddings (raw or L2-normalized, both work).

        Returns
        -------
        np.ndarray, shape (N, tgt_dim)
            L2-normalized aligned embeddings ready for cosine similarity search.
        """
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]

        v_c = vecs - self._src_mean
        if self._W_svd is not None:
            v_c = v_c @ self._W_svd
        v_c = l2_normalize(v_c)
        out = v_c @ self._R
        return l2_normalize(out)

    def __repr__(self) -> str:
        return (
            f"PureZeroShotAdapter("
            f"src_dim={self.src_dim}, tgt_dim={self.tgt_dim})"
        )
