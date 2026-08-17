"""AnnData preparation helpers for scGPT cell annotation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

from .preprocess import Preprocessor
from .tokenizer import GeneVocab, tokenize_and_pad_batch


@dataclass
class AnnotationTensors:
    """Tokenized train and validation tensors for cell annotation."""

    train: dict[str, torch.Tensor]
    validation: dict[str, torch.Tensor]
    label_names: list[str]
    gene_column: str
    data_is_raw: bool


class CellAnnotationDataset(torch.utils.data.Dataset):
    """Dictionary-style tensor dataset consumed by scGPT training loops."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors["gene_ids"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def resolve_gene_column(adata: AnnData, gene_column: Optional[str] = None) -> str:
    """Resolve a gene symbol column, using the variable index as a fallback."""
    if gene_column == "index":
        adata.var["index"] = adata.var_names.astype(str)
        return "index"
    if gene_column is not None:
        if gene_column not in adata.var:
            raise KeyError(f"Gene column {gene_column!r} is not present in adata.var")
        return gene_column
    for candidate in (
        "Gene Symbol",
        "feature_name",
        "gene_name",
        "gene_symbols",
        "symbol",
    ):
        if candidate in adata.var:
            return candidate
    adata.var["index"] = adata.var_names.astype(str)
    return "index"


def infer_data_is_raw(adata: AnnData) -> bool:
    """Return whether the expression matrix looks like non-negative raw counts."""
    matrix = adata.X
    values = matrix.data if issparse(matrix) else np.asarray(matrix).reshape(-1)
    if values.size == 0:
        return True
    if not np.isfinite(values).all() or values.min() < 0:
        return False
    return bool(np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6))


def prepare_cell_annotation_data(
    adata: AnnData,
    vocab: GeneVocab,
    *,
    label_column: str,
    gene_column: Optional[str] = None,
    n_hvg: int = 1200,
    n_bins: int = 51,
    max_length: int = 1201,
    validation_fraction: float = 0.1,
    seed: int = 42,
    data_is_raw: Optional[bool] = None,
) -> AnnotationTensors:
    """Match genes, preprocess expression values, split cells, and tokenize them."""
    if label_column not in adata.obs:
        raise KeyError(f"Label column {label_column!r} is not present in adata.obs")
    adata = adata.copy()
    gene_column = resolve_gene_column(adata, gene_column)
    genes = adata.var[gene_column].astype(str)
    in_vocab = np.asarray([gene in vocab for gene in genes], dtype=bool)
    if not in_vocab.any():
        raise ValueError("No input genes matched the selected scGPT vocabulary")
    adata = adata[:, in_vocab].copy()
    adata.var["gene_name"] = adata.var[gene_column].astype(str).to_numpy()
    if data_is_raw is None:
        data_is_raw = infer_data_is_raw(adata)
    if not data_is_raw:
        min_cells = min(3, adata.n_obs)
        if issparse(adata.X):
            expressed_cells = np.asarray(adata.X.getnnz(axis=0)).reshape(-1)
        else:
            expressed_cells = np.count_nonzero(np.asarray(adata.X), axis=0)
        expressed = expressed_cells >= min_cells
        if not expressed.any():
            raise ValueError(
                f"No genes are expressed in at least {min_cells} input cells"
            )
        adata = adata[:, expressed].copy()

    subset_hvg = n_hvg if n_hvg and n_hvg < adata.n_vars else False
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3 if data_is_raw else False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=data_is_raw,
        result_log1p_key="X_log1p",
        subset_hvg=subset_hvg,
        hvg_flavor="cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata)

    matrix = adata.layers["X_binned"]
    if issparse(matrix):
        matrix = matrix.toarray()
    gene_ids = np.asarray(vocab(adata.var["gene_name"].tolist()), dtype=np.int64)
    tokenized = tokenize_and_pad_batch(
        matrix,
        gene_ids,
        max_len=max_length,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=True,
    )

    labels = adata.obs[label_column].astype("category")
    label_ids = labels.cat.codes.to_numpy(dtype=np.int64)
    indices = np.arange(adata.n_obs)
    counts = np.bincount(label_ids)
    stratify = label_ids if counts.size and counts.min() >= 2 else None
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    def select(selected: np.ndarray) -> dict[str, torch.Tensor]:
        return {
            "gene_ids": tokenized["genes"][selected],
            "values": tokenized["values"][selected].float(),
            "labels": torch.from_numpy(label_ids[selected]).long(),
        }

    return AnnotationTensors(
        train=select(train_indices),
        validation=select(validation_indices),
        label_names=[str(name) for name in labels.cat.categories],
        gene_column=gene_column,
        data_is_raw=data_is_raw,
    )


__all__ = [
    "AnnotationTensors",
    "CellAnnotationDataset",
    "infer_data_is_raw",
    "prepare_cell_annotation_data",
    "resolve_gene_column",
]
