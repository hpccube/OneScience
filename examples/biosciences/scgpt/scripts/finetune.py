"""Fine-tune scGPT for cell-type classification from an AnnData dataset."""

import argparse
import json
import os
from pathlib import Path

import scanpy as sc
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.scgpt import (
    CellAnnotationDataset,
    prepare_cell_annotation_data,
)
from onescience.datapipes.scgpt.tokenizer import GeneVocab
from onescience.models.scgpt import load_model_and_vocab, load_model_config
from onescience.utils.scgpt import (
    distributed_barrier,
    finalize_distributed,
    initialize_distributed,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--label-column", default="Celltype")
    parser.add_argument("--gene-column")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--n-hvg", type=int, default=1200)
    parser.add_argument("--max-length", type=int, default=1201)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-encoder", action="store_true")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-is-raw",
        dest="data_is_raw",
        action="store_true",
        help="Treat adata.X as raw integer counts.",
    )
    data_group.add_argument(
        "--data-is-normalized",
        dest="data_is_raw",
        action="store_false",
        help="Treat adata.X as an already processed expression matrix.",
    )
    parser.set_defaults(data_is_raw=None)
    return parser.parse_args()


def evaluate(model, loader, vocab, device, amp_enabled, distributed):
    model.eval()
    predictions = []
    labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch in loader:
            gene_ids = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            targets = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(
                    gene_ids,
                    values,
                    src_key_padding_mask=gene_ids.eq(vocab["<pad>"]),
                    CLS=True,
                )["cls_output"]
                loss = criterion(logits, targets)
            total_loss += loss.item() * targets.numel()
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.cpu().tolist())
    if distributed:
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (predictions, labels, total_loss))
        predictions = [value for item in gathered for value in item[0]]
        labels = [value for item in gathered for value in item[1]]
        total_loss = sum(item[2] for item in gathered)
    return {
        "loss": total_loss / max(len(labels), 1),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def read_cells(path: Path, maximum: int | None):
    backed = sc.read_h5ad(path, backed="r")
    try:
        stop = min(backed.n_obs, maximum) if maximum is not None else backed.n_obs
        return backed[:stop].to_memory()
    finally:
        backed.file.close()


def main() -> None:
    args = parse_args()
    context = initialize_distributed(args.device)
    set_seed(args.seed + context.rank)
    device = context.device
    amp_enabled = device.type == "cuda"
    cache_file = args.output_dir / (
        f".preprocessed-{os.environ.get('TORCHELASTIC_RUN_ID', 'single')}.pt"
    )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        vocab = GeneVocab.from_file(args.model_dir / "vocab.json")
        for token in ("<pad>", "<cls>", "<eoc>"):
            if token not in vocab:
                vocab.append_token(token)
        vocab.set_default_index(vocab["<pad>"])
        if context.is_main:
            adata = read_cells(args.data_file, args.max_cells)
            tensors = prepare_cell_annotation_data(
                adata,
                vocab,
                label_column=args.label_column,
                gene_column=args.gene_column,
                n_hvg=args.n_hvg,
                n_bins=load_model_config(args.model_dir).get("n_bins", 51),
                max_length=args.max_length,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                data_is_raw=args.data_is_raw,
            )
            if context.enabled:
                torch.save(tensors, cache_file)
        if context.enabled:
            distributed_barrier(context)
            if not context.is_main:
                tensors = torch.load(cache_file, map_location="cpu", weights_only=False)
            distributed_barrier(context)
            if context.is_main:
                cache_file.unlink(missing_ok=True)
        if context.is_main:
            interpretation = "raw counts" if tensors.data_is_raw else "normalized"
            print(
                f"Input expression interpreted as {interpretation}; "
                f"training on {context.world_size} device(s)",
                flush=True,
            )

        model, vocab, model_config = load_model_and_vocab(
            args.model_dir,
            n_cls=len(tensors.label_names),
            device=device,
            use_fast_transformer=False,
            load_weights=context.is_main,
        )
        if args.freeze_encoder:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.cls_decoder.parameters():
                parameter.requires_grad = True
        if context.enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=not args.freeze_encoder,
            )

        train_dataset = CellAnnotationDataset(tensors.train)
        validation_dataset = CellAnnotationDataset(tensors.validation)
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=args.seed,
            )
            if context.enabled
            else None
        )
        validation_sampler = (
            list(range(context.rank, len(validation_dataset), context.world_size))
            if context.enabled
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            sampler=validation_sampler,
            shuffle=False,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        criterion = nn.CrossEntropyLoss()
        best_metrics = None
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            for batch in train_loader:
                gene_ids = batch["gene_ids"].to(device)
                values = batch["values"].to(device)
                targets = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(
                        gene_ids,
                        values,
                        src_key_padding_mask=gene_ids.eq(vocab["<pad>"]),
                        CLS=True,
                    )["cls_output"]
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                if args.max_steps is not None and global_step >= args.max_steps:
                    break

            evaluation_model = model.module if context.enabled else model
            metrics = evaluate(
                evaluation_model,
                validation_loader,
                vocab,
                device,
                amp_enabled,
                context.enabled,
            )
            metrics.update({"epoch": epoch, "step": global_step})
            if context.is_main:
                print(json.dumps(metrics, sort_keys=True))
            if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
                best_metrics = metrics
                if context.is_main:
                    torch.save(
                        evaluation_model.state_dict(), args.output_dir / "best_model.pt"
                    )
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if context.is_main:
            output_config = dict(model_config)
            output_config.update(
                {
                    "n_cls": len(tensors.label_names),
                    "label_names": tensors.label_names,
                    "gene_column": tensors.gene_column,
                    "data_is_raw": tensors.data_is_raw,
                    "source_model": str(args.model_dir.resolve()),
                }
            )
            (args.output_dir / "args.json").write_text(
                json.dumps(output_config, indent=2, sort_keys=True), encoding="utf-8"
            )
            (args.output_dir / "metrics.json").write_text(
                json.dumps(best_metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
            vocab.save_json(args.output_dir / "vocab.json")
            print(f"Saved fine-tuned checkpoint to {args.output_dir}")
        distributed_barrier(context)
    finally:
        if context.is_main:
            cache_file.unlink(missing_ok=True)
        finalize_distributed(context)


if __name__ == "__main__":
    main()
