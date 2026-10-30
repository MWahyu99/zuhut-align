import numpy as np


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D array."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms
