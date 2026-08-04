#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

LOCAL_TRAINING_ENTRYPOINT = "local_sft_no_trl"

DEFAULT_PARQUET_DIR = os.path.join(os.getenv("ONESCIENCE_DATASETS_DIR", ""), "medgemma", "medqa")


def main():
    parser = argparse.ArgumentParser(description="Local SFT tuning entry point for MedGemma on MedQA.")
    parser.add_argument("--model_path", default=os.getenv("MEDGEMMA_MODEL_PATH"))
    parser.add_argument("--dataset_name", default=os.getenv("MEDQA_DATASET_NAME", "openlifescienceai/medqa"))
    parser.add_argument("--parquet_dir", default=os.getenv("MEDQA_PARQUET_DIR") or os.getenv("PARQUET_DIR") or DEFAULT_PARQUET_DIR)
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./medgemma_lora_outputs"))
    parser.add_argument("--max_train_samples", type=int, default=256)
    parser.add_argument("--max_eval_samples", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=int(os.getenv("MAX_SEQ_LENGTH", "1024")))
    parser.add_argument("--use_lora", action="store_true", help="Enable PEFT LoRA. Disabled by default to avoid broken bitsandbytes ROCm imports.")
    parser.add_argument("--dry_run", action="store_true", help="Only validate arguments and imports.")
    args = parser.parse_args()

    print(f"Training entrypoint: {LOCAL_TRAINING_ENTRYPOINT}")

    if args.dry_run:
        print("Dry run OK")
        print(args)
        return

    if not args.model_path:
        raise ValueError("Set MEDGEMMA_MODEL_PATH or pass --model_path for tuning.")

    import datasets
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    def load_medqa_dataset():
        if args.parquet_dir:
            args.parquet_dir = os.path.abspath(args.parquet_dir)
            if not os.path.isdir(args.parquet_dir):
                raise FileNotFoundError(f"Local MedQA parquet directory does not exist: {args.parquet_dir}")
            print(f"Loading MedQA dataset from local Parquet directory: {args.parquet_dir}")
            data_files = {
                "train": os.path.join(args.parquet_dir, "train-*.parquet"),
                "test": os.path.join(args.parquet_dir, "test-*.parquet"),
                "validation": os.path.join(args.parquet_dir, "dev-*.parquet"),
            }
            return datasets.load_dataset("parquet", data_files=data_files)
        raise ValueError("No local parquet_dir was provided. Set PARQUET_DIR or MEDQA_PARQUET_DIR; online Hub loading is disabled for this local script.")

    def format_training_text(row):
        data = row["data"]
        question = data["Question"]
        options = data["Options"]
        answer = data["Correct Option"]
        prompt = (
            "Answer the given medical multiple-choice question. Think briefly, then give the final answer as a single letter.\n"
            f"Question: {question}\n"
            f"(A) {options['A']} (B) {options['B']} (C) {options['C']} (D) {options['D']}\n"
            f"Final Answer: ({answer})"
        )
        return {"text": prompt}

    dataset = load_medqa_dataset()
    train_split = dataset["train"]
    eval_split = dataset.get("validation") or dataset.get("dev") or dataset.get("test")
    train = train_split.select(range(min(args.max_train_samples, len(train_split)))).map(format_training_text, remove_columns=train_split.column_names)
    eval_ds = eval_split.select(range(min(args.max_eval_samples, len(eval_split)))).map(format_training_text, remove_columns=eval_split.column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    train = train.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        for _, param in model.named_parameters():
            param.requires_grad = False
        trainable_keywords = ("lm_head", "embed_tokens", "norm")
        trainable = 0
        total = 0
        for name, param in model.named_parameters():
            total += param.numel()
            if any(key in name for key in trainable_keywords):
                param.requires_grad = True
                trainable += param.numel()
        print(f"LoRA disabled; trainable parameters: {trainable:,} / {total:,}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved training output to: {os.path.abspath(args.output_dir)}")
    print("Saved files:")
    for filename in sorted(os.listdir(args.output_dir)):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()