#!/usr/bin/env python3
import argparse
import copy
import json
import os
import tempfile

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

LOCAL_TRAINING_ENTRYPOINT = "local_sft_no_trl"
DEFAULT_PARQUET_DIR = os.path.join(
    os.getenv("ONESCIENCE_DATASETS_DIR", ""), "medgemma", "medqa"
)


def main():
    parser = argparse.ArgumentParser(
        description="Local SFT tuning entry point for MedGemma on MedQA."
    )
    parser.add_argument("--model_path", default=os.getenv("MEDGEMMA_MODEL_PATH"))
    parser.add_argument(
        "--dataset_name",
        default=os.getenv("MEDQA_DATASET_NAME", "openlifescienceai/medqa"),
    )
    parser.add_argument(
        "--parquet_dir",
        default=os.getenv("MEDQA_PARQUET_DIR")
        or os.getenv("PARQUET_DIR")
        or DEFAULT_PARQUET_DIR,
    )
    parser.add_argument(
        "--output_dir", default=os.getenv("OUTPUT_DIR", "./medgemma_lora_outputs")
    )
    parser.add_argument("--max_train_samples", type=int, default=256)
    parser.add_argument("--max_eval_samples", type=int, default=64)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=int(os.getenv("MAX_SEQ_LENGTH", "1024")),
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--deepspeed", default=os.getenv("DEEPSPEED_CONFIG"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Training entrypoint: {LOCAL_TRAINING_ENTRYPOINT}", flush=True)
    print(f"DeepSpeed config: {args.deepspeed or 'disabled'}", flush=True)

    if args.dry_run:
        print("Dry run OK")
        print(args)
        return
    if not args.model_path:
        raise ValueError("Set MEDGEMMA_MODEL_PATH or pass --model_path for tuning.")

    import datasets
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    deepspeed_init_path = None
    deepspeed_runtime_path = None
    hf_deepspeed_config = None
    if args.deepspeed:
        from transformers.integrations import HfDeepSpeedConfig

        with open(args.deepspeed, encoding="utf-8") as config_file:
            base_config = json.load(config_file)
        micro_batch = int(base_config["train_micro_batch_size_per_gpu"])
        accumulation = int(base_config["gradient_accumulation_steps"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch_size = micro_batch * accumulation * world_size

        runtime_config = copy.deepcopy(base_config)
        runtime_config["train_batch_size"] = effective_batch_size
        descriptor, deepspeed_runtime_path = tempfile.mkstemp(
            prefix="medgemma_zero3_runtime_", suffix=".json"
        )
        os.close(descriptor)
        with open(deepspeed_runtime_path, "w", encoding="utf-8") as config_file:
            json.dump(runtime_config, config_file)
        args.deepspeed = deepspeed_runtime_path

        init_config = copy.deepcopy(base_config)
        init_config["train_batch_size"] = micro_batch * accumulation
        descriptor, deepspeed_init_path = tempfile.mkstemp(
            prefix="medgemma_zero3_init_", suffix=".json"
        )
        os.close(descriptor)
        with open(deepspeed_init_path, "w", encoding="utf-8") as config_file:
            json.dump(init_config, config_file)
        hf_deepspeed_config = HfDeepSpeedConfig(deepspeed_init_path)
        print(
            f"DeepSpeed world size: {world_size}; effective train batch size: "
            f"{effective_batch_size}",
            flush=True,
        )

    def load_medqa_dataset():
        parquet_dir = os.path.abspath(args.parquet_dir)
        if not os.path.isdir(parquet_dir):
            raise FileNotFoundError(
                f"Local MedQA parquet directory does not exist: {parquet_dir}"
            )
        print(f"Loading MedQA dataset from: {parquet_dir}", flush=True)
        return datasets.load_dataset(
            "parquet",
            data_files={
                "train": os.path.join(parquet_dir, "train-*.parquet"),
                "test": os.path.join(parquet_dir, "test-*.parquet"),
                "validation": os.path.join(parquet_dir, "dev-*.parquet"),
            },
        )

    def format_training_text(row):
        data = row["data"]
        options = data["Options"]
        return {
            "text": (
                "Answer the given medical multiple-choice question. Think briefly, "
                "then give the final answer as a single letter.\n"
                f"Question: {data['Question']}\n"
                f"(A) {options['A']} (B) {options['B']} "
                f"(C) {options['C']} (D) {options['D']}\n"
                f"Final Answer: ({data['Correct Option']})"
            )
        }

    dataset = load_medqa_dataset()
    train_split = dataset["train"]
    eval_split = dataset.get("validation") or dataset.get("dev") or dataset["test"]
    train = train_split.select(
        range(min(args.max_train_samples, len(train_split)))
    ).map(format_training_text, remove_columns=train_split.column_names)
    eval_dataset = eval_split.select(
        range(min(args.max_eval_samples, len(eval_split)))
    ).map(format_training_text, remove_columns=eval_split.column_names)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train = train.map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    else:
        raise ValueError("The 27B multi-GPU example requires USE_LORA=1.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    if trainer.is_world_process_zero():
        print("Training completed successfully.", flush=True)

    if deepspeed_init_path:
        os.unlink(deepspeed_init_path)
    if deepspeed_runtime_path:
        os.unlink(deepspeed_runtime_path)
    del hf_deepspeed_config


if __name__ == "__main__":
    main()
