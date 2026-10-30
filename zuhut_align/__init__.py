"""
zuhut-align
===========
Cross-model embedding alignment SDK.

Bridge any two embedding models with zero-shot alignment or
self-supervised contrastive fine-tuning.

Quick start
-----------
>>> from zuhut_align import ZUHUTClient
>>> client = ZUHUTClient()
>>> adapter = client.create_adapter(src_dim=1024)
>>> aligned = adapter.transform(my_vectors)   # shape (N, 384)

Tiers
-----
Level 1 — ZeroShot (free)
    No training data. Works out-of-the-box.
    client.create_adapter(src_dim=..., tier='zeroshot')

Level 2 — PhaseMLP (paid)
    Self-supervised on your own corpus (~5 min, 200-500 docs).
    91% of same-model retrieval ceiling. Requires api_key.
    client.create_adapter(src_dim=..., tier='phasemlp')

Level 3 — Rust Engine (enterprise)
    10–50x faster inference. Requires license_key.
    client.create_adapter(src_dim=..., tier='engine')
"""

from .client import ZUHUTClient
from .adapters.zero_shot import PureZeroShotAdapter
from .mlp.phase_mlp import PhaseMLP, PhaseMLP_Trainer, PhaseMLP_Adapter
from .utils.normalize import l2_normalize

__version__ = "0.1.2"
__author__  = "ZUHUT"
__email__   = "contact@zuhut.ai"

__all__ = [
    # Main entry point
    "ZUHUTClient",
    # Level 1 — ZeroShot
    "PureZeroShotAdapter",
    # Level 2 — PhaseMLP
    "PhaseMLP",
    "PhaseMLP_Trainer",
    "PhaseMLP_Adapter",
    # Utilities
    "l2_normalize",
]
