"""
zuhut_align.client
==================
Main entry point for the ZUHUT Align SDK.

Usage
-----
Free tier (zero-shot, no API key needed):
    from zuhut_align import ZUHUTClient
    client = ZUHUTClient()
    adapter = client.create_adapter(src_dim=1024)
    aligned = adapter.transform(my_vectors)

Paid tier (PhaseMLP, requires training corpus):
    client = ZUHUTClient(api_key="zuhut-...")
    adapter = client.create_adapter(src_dim=1024, tier="phasemlp")
    adapter.train(corpus_texts, embedder=model.encode)
    aligned = adapter.transform(query_vecs)

Enterprise tier (Rust engine, requires license):
    client = ZUHUTClient(api_key="zuhut-...", license_key="enterprise-...")
    adapter = client.create_adapter(src_dim=1024, tier="engine")
    aligned = adapter.transform(query_vecs)
"""

from __future__ import annotations

from typing import Optional, Literal

from .adapters.zero_shot import PureZeroShotAdapter
from .mlp.phase_mlp import PhaseMLP_Trainer, PhaseMLP_Adapter
from .engine import EngineAdapter

__all__ = ["ZUHUTClient"]

# Tier literals
Tier = Literal["zeroshot", "phasemlp", "engine"]


class ZUHUTClient:
    """
    ZUHUT Align client — bridge any two embedding models, zero-shot.

    Parameters
    ----------
    api_key : str, optional
        Required for Level 2 (PhaseMLP) and above.
        Not required for Level 1 (ZeroShot, free).
    license_key : str, optional
        Required for Level 3 (Rust Engine, enterprise).
    tgt_dim : int
        Target embedding dimension. Default 384.
    n_anchors : int
        Number of synthetic anchors for zero-shot alignment. Default 250.
    seed : int
        Random seed for reproducibility. Default 42.

    Example
    -------
    >>> from zuhut_align import ZUHUTClient
    >>> client = ZUHUTClient()
    >>> adapter = client.create_adapter(src_dim=1024)
    >>> query_aligned = adapter.transform(raw_query_vec)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        license_key: Optional[str] = None,
        tgt_dim: int = 384,
        n_anchors: int = 250,
        seed: int = 42,
    ):
        self._api_key = api_key
        self._license_key = license_key
        self._tgt_dim = tgt_dim
        self._n_anchors = n_anchors
        self._seed = seed

    def create_adapter(
        self,
        src_dim: int,
        tier: Tier = "zeroshot",
        *,
        # PhaseMLP-specific kwargs
        hidden: int = 128,
        dropout: float = 0.3,
        epochs: int = 300,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        temperature: float = 0.07,
        n_augment: int = 5,
        verbose: bool = True,
    ):
        """
        Create an embedding adapter for the specified tier.

        Parameters
        ----------
        src_dim : int
            Source model dimensionality (e.g. 1024 for bge-large-en-v1.5).
        tier : {"zeroshot", "phasemlp", "engine"}
            Which tier to use:
            - "zeroshot" : Level 1 — free, no training required
            - "phasemlp" : Level 2 — requires api_key + training corpus
            - "engine"   : Level 3 — requires license_key (enterprise)
        hidden : int
            PhaseMLP hidden dim (tier="phasemlp" only). Default 128.
        dropout : float
            PhaseMLP dropout rate (tier="phasemlp" only). Default 0.3.
        epochs : int
            Training epochs (tier="phasemlp" only). Default 300.
        lr : float
            Learning rate (tier="phasemlp" only). Default 3e-4.
        weight_decay : float
            AdamW weight decay (tier="phasemlp" only). Default 1e-3.
        temperature : float
            InfoNCE temperature (tier="phasemlp" only). Default 0.07.
        n_augment : int
            Pseudo-pairs per doc (tier="phasemlp" only). Default 5.
        verbose : bool
            Print training progress (tier="phasemlp" only). Default True.

        Returns
        -------
        adapter
            - tier="zeroshot" : PureZeroShotAdapter
              Call: adapter.transform(vecs)
            - tier="phasemlp" : PhaseMLP_Trainer (untrained)
              Call: adapter.train(texts, embedder), then adapter.transform(vecs)
            - tier="engine"   : EngineAdapter (enterprise)

        Raises
        ------
        ValueError
            If the requested tier requires credentials not provided.
        """
        tier = tier.lower().strip()

        if tier == "zeroshot":
            return self._create_zeroshot(src_dim)

        elif tier == "phasemlp":
            raise NotImplementedError(
                "ZUHUT PhaseMLP (Level 2) is coming soon.\n"
                "\n"
                "Join the waitlist to get early access:\n"
                "  https://github.com/zuhut-ai/zuhut-align\n"
                "\n"
                "Available now:\n"
                "  Level 1 — ZeroShot (free): client.create_adapter(tier='zeroshot')"
            )

        elif tier == "engine":
            self._require_license_key()
            return EngineAdapter(license_key=self._license_key)

        else:
            raise ValueError(
                f"Unknown tier: {tier!r}. "
                f"Choose one of: 'zeroshot', 'phasemlp', 'engine'"
            )

    def _create_zeroshot(self, src_dim: int) -> PureZeroShotAdapter:
        return PureZeroShotAdapter(
            src_dim=src_dim,
            tgt_dim=self._tgt_dim,
            n_anchors=self._n_anchors,
            seed=self._seed,
        )

    def _create_phasemlp(self, src_dim: int, **kwargs) -> PhaseMLP_Trainer:
        linear = self._create_zeroshot(src_dim)
        return PhaseMLP_Trainer(
            linear_adapter=linear,
            dim=self._tgt_dim,
            **kwargs,
        )

    def _require_api_key(self, tier_name: str):
        if not self._api_key:
            raise ValueError(
                f"The '{tier_name}' tier requires an API key.\n"
                f"Create a ZUHUTClient with: ZUHUTClient(api_key='zuhut-...')\n"
                f"\n"
                f"Or use the free tier: client.create_adapter(src_dim=..., tier='zeroshot')"
            )

    def _require_license_key(self):
        if not self._license_key:
            raise ValueError(
                "The 'engine' tier requires an enterprise license key.\n"
                "Contact: contact@zuhut.ai"
            )

    def __repr__(self) -> str:
        tier = "enterprise" if self._license_key else \
               "paid" if self._api_key else "free"
        return f"ZUHUTClient(tier={tier!r}, tgt_dim={self._tgt_dim})"
