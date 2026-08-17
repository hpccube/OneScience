"""Generate scGPT cell embeddings from a local AnnData file."""

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import torch.distributed as dist

from onescience.utils.scgpt import (
    embed_data,
    contiguous_shard_bounds,
    distributed_barrier,
    finalize_distributed,
    initialize_distributed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/embeddings.h5ad"))
    parser.add_argument("--gene-column")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-fast-transformer", action="store_true")
    return parser.parse_args()


def _read_cells(path: Path, start: int, stop: int):
    backed = sc.read_h5ad(path, backed="r")
    try:
        return backed[start:stop].to_memory()
    finally:
        backed.file.close()


def _cell_count(path: Path, maximum: int | None) -> int:
    backed = sc.read_h5ad(path, backed="r")
    try:
        return min(backed.n_obs, maximum) if maximum is not None else backed.n_obs
    finally:
        backed.file.close()


def main() -> None:
    args = parse_args()
    context = initialize_distributed(args.device)
    try:
        total_cells = _cell_count(args.data_file, args.max_cells)
        if total_cells < context.world_size:
            raise ValueError(
                f"Input has {total_cells} cells but {context.world_size} devices were detected"
            )
        start, stop = contiguous_shard_bounds(
            total_cells, context.rank, context.world_size
        )
        if context.is_main:
            print(
                f"Embedding {total_cells} cells on {context.world_size} device(s)",
                flush=True,
            )
        adata = _read_cells(args.data_file, start, stop)
        embedded = embed_data(
            adata,
            args.model_dir,
            gene_col=args.gene_column,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=context.device,
            use_fast_transformer=args.use_fast_transformer,
            return_new_adata=False,
            show_progress=context.is_main,
        )
        payload = (start, np.asarray(embedded.obsm["X_scGPT"]))
        if context.enabled:
            gathered = [None] * context.world_size if context.is_main else None
            dist.gather_object(payload, gathered, dst=0)
        else:
            gathered = [payload]

        if context.is_main:
            gathered.sort(key=lambda item: item[0])
            embeddings = np.concatenate([item[1] for item in gathered], axis=0)
            output_adata = _read_cells(args.data_file, 0, total_cells)
            output_adata = output_adata[:, embedded.var_names].copy()
            output_adata.obsm["X_scGPT"] = embeddings
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_adata.write_h5ad(args.output)
            print(f"Saved {output_adata.n_obs} embeddings to {args.output}")
        distributed_barrier(context)
    finally:
        finalize_distributed(context)


if __name__ == "__main__":
    main()
