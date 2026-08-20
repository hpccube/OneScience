"""Pretrain a Geneformer masked-language model on a tokenized cell corpus."""

import argparse
import pickle
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments

from onescience.datapipes.geneformer.resources import TOKEN_DICTIONARY_FILE_30M
from onescience.utils.geneformer import GeneformerPretrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--lengths-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrain"))
    parser.add_argument("--token-dictionary", type=Path, default=TOKEN_DICTIONARY_FILE_30M)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Geneformer pretraining requires a visible CUDA/DTK device")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(args.data_file))
    lengths_file = args.lengths_file
    if args.max_cells is not None:
        cell_count = min(args.max_cells, len(dataset))
        dataset = dataset.select(range(cell_count))
        lengths_file = args.output_dir / "subset_lengths.pkl"
        with lengths_file.open("wb") as handle:
            pickle.dump(list(dataset["length"]), handle)

    with args.token_dictionary.open("rb") as handle:
        token_dictionary = pickle.load(handle)

    config = BertConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        attention_probs_dropout_prob=0.02,
        hidden_dropout_prob=0.02,
        intermediate_size=args.hidden_size * 2,
        hidden_act="relu",
        max_position_embeddings=args.max_position_embeddings,
        model_type="bert",
        num_attention_heads=args.num_heads,
        pad_token_id=token_dictionary["<pad>"],
        vocab_size=len(token_dictionary),
    )
    model = BertForMaskedLM(config)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        logging_dir=str(args.output_dir / "logs"),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        group_by_length=True,
        length_column_name="length",
        save_strategy="steps",
        save_steps=max(args.max_steps, 1) if args.max_steps > 0 else 10_000,
        logging_steps=1,
        report_to="none",
        seed=args.seed,
        bf16=args.bf16,
        overwrite_output_dir=args.overwrite_output_dir,
    )
    trainer = GeneformerPretrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        example_lengths_file=str(lengths_file),
        token_dictionary=token_dictionary,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "model"))
    print(f"Saved pretrained model to {args.output_dir / 'model'}")


if __name__ == "__main__":
    main()
