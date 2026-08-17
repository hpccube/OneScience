import os
from typing import Optional, Union

import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist
from anndata import AnnData
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from onescience.datapipes.scgpt import DataCollator, resolve_gene_column
from onescience.models.scgpt import load_model_and_vocab
from onescience.utils.scgpt.distributed import broadcast_module

from .logging import logger

PathLike = Union[str, os.PathLike]


def get_batch_cell_embeddings(
    adata,
    cell_embedding_mode: str = "cls",
    model=None,
    vocab=None,
    max_length=1200,
    batch_size=64,
    model_configs=None,
    gene_ids=None,
    use_batch_labels=False,
    show_progress=True,
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        cell_embedding_mode (str): The mode to get the cell embeddings. Defaults to "cls".
        model (torch.nn.Module, optional): The model. Defaults to None.
        vocab (GeneVocab, optional): The vocabulary. Defaults to None.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        model_configs (dict, optional): The model configurations. Defaults to None.
        gene_ids (np.ndarray, optional): The gene vocabulary ids. Defaults to None.
        use_batch_labels (bool): Whether to use batch labels. Defaults to False.

    Returns:
        np.ndarray: The cell embeddings.
    """

    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.toarray()
    )

    # gene vocabulary ids
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if use_batch_labels:
        batch_ids = np.array(adata.obs["batch_id"].tolist())

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, count_matrix, gene_ids, batch_ids=None):
            self.count_matrix = count_matrix
            self.gene_ids = gene_ids
            self.batch_ids = batch_ids

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = self.gene_ids[nonzero_idx]
            # append <cls> token at the beginning
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs["pad_value"])
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            output = {
                "id": idx,
                "genes": genes,
                "expressions": values,
            }
            if self.batch_ids is not None:
                output["batch_labels"] = self.batch_ids[idx]
            return output

    if cell_embedding_mode == "cls":
        dataset = Dataset(
            count_matrix, gene_ids, batch_ids if use_batch_labels else None
        )
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs["pad_token"]],
            pad_value=model_configs["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), batch_size, 8),
            pin_memory=True,
        )

        device = next(model.parameters()).device
        cell_embeddings = np.zeros(
            (len(dataset), model_configs["embsize"]), dtype=np.float32
        )
        amp_enabled = device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            count = 0
            for data_dict in tqdm(
                data_loader, desc="Embedding cells", disable=not show_progress
            ):
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[model_configs["pad_token"]]
                )
                embeddings = model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels
                    else None,
                )

                embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)
        norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
        cell_embeddings = cell_embeddings / np.clip(norms, a_min=1e-12, a_max=None)
    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")
    return cell_embeddings


def embed_data(
    adata_or_file: Union[AnnData, PathLike],
    model_dir: PathLike,
    gene_col: Optional[str] = None,
    max_length=1200,
    batch_size=64,
    obs_to_save: Optional[list] = None,
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = True,
    return_new_adata: bool = False,
    show_progress: bool = True,
) -> AnnData:
    """
    Preprocess anndata and embed the data using the model.

    Args:
        adata_or_file (Union[AnnData, PathLike]): The AnnData object or the path to the
            AnnData object.
        model_dir (PathLike): The path to the model directory.
        gene_col (str): The column in adata.var that contains the gene names. Common
            columns are auto-detected when omitted, then var_names are used as fallback.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        obs_to_save (Optional[list]): The list of obs columns to save in the output adata.
            Useful for retaining meta data to output. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cuda".
        use_fast_transformer (bool): Whether to use flash-attn. Defaults to True.
        return_new_adata (bool): Whether to return a new AnnData object. If False, will
            add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".

    Returns:
        AnnData: The AnnData object with the cell embeddings.
    """
    if isinstance(adata_or_file, AnnData):
        adata = adata_or_file
    else:
        adata = sc.read_h5ad(adata_or_file)

    if isinstance(obs_to_save, str):
        assert obs_to_save in adata.obs, f"obs_to_save {obs_to_save} not in adata.obs"
        obs_to_save = [obs_to_save]

    adata = adata.copy()
    gene_col = resolve_gene_column(adata, gene_col)
    device = torch.device(device)
    distributed = dist.is_available() and dist.is_initialized()
    model, vocab, model_configs = load_model_and_vocab(
        model_dir,
        device=device,
        use_fast_transformer=use_fast_transformer,
        load_weights=not distributed or dist.get_rank() == 0,
    )
    if distributed:
        broadcast_module(model)
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    genes = adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    model.eval()

    # get cell embeddings
    cell_embeddings = get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
        show_progress=show_progress,
    )

    if return_new_adata:
        obs_df = adata.obs[obs_to_save] if obs_to_save is not None else None
        return sc.AnnData(X=cell_embeddings, obs=obs_df, dtype="float32")

    adata.obsm["X_scGPT"] = cell_embeddings
    return adata
