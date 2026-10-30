"""
ZUHUT Align — Cross-model embedding alignment SDK.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="zuhut-align",
    version="0.1.2",
    author="ZUHUT",
    author_email="contact@zuhut.ai",
    description="Cross-model embedding alignment: bridge any two embedding models, zero-shot.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zuhut-ai/zuhut-align",
    packages=find_packages(),
    python_requires=">=3.9",

    # Core dependencies (Level 1 — ZeroShot, free)
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],

    # Optional extras
    extras_require={
        # Level 2 — PhaseMLP training + inference
        "mlp": [
            "torch>=2.0",
            "sentence-transformers>=2.7",
        ],
        # Level 2 + Qdrant integration example
        "qdrant": [
            "torch>=2.0",
            "sentence-transformers>=2.7",
            "qdrant-client>=1.9",
        ],
        # Full install (everything except engine)
        "full": [
            "torch>=2.0",
            "sentence-transformers>=2.7",
            "qdrant-client>=1.9",
        ],
        # Development
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "ruff",
            "mypy",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    keywords=[
        "embeddings", "alignment", "cross-model", "retrieval",
        "vector search", "semantic search", "sentence transformers",
        "zero-shot", "contrastive learning",
    ],

    project_urls={
        "Documentation": "https://docs.zuhut.ai",
        "Homepage":      "https://zuhut.ai",
        "Bug Reports":   "https://github.com/zuhut-ai/zuhut-align/issues",
        "Source":        "https://github.com/zuhut-ai/zuhut-align",
    },
)
