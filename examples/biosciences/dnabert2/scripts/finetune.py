"""LoRA fine-tuning entry point for DNABERT-2."""

from train import main


if __name__ == "__main__":
    main(default_lora=True)
