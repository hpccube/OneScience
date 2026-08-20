"""Geneformer-specific model structures.

The pretrained V1 and V2 encoders use Hugging Face BERT directly. Geneformer adds
a custom network only for multi-task cell classification.
"""

from .mtl.model import AttentionPool, GeneformerMultiTask

__all__ = ["AttentionPool", "GeneformerMultiTask"]
