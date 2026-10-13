#!/usr/bin/env python3
import argparse

from medgemma_script_utils import LocalMedGemmaRunner, add_common_args, load_image, write_outputs


def main():
    parser = argparse.ArgumentParser(description="Quick start MedGemma local inference using downloaded Model Garden assets.")
    add_common_args(parser, multimodal=True)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--prompt", default="Describe this X-ray.")
    args = parser.parse_args()

    images = [load_image(args.image_path)] if args.image_path else []
    runner = LocalMedGemmaRunner(args.model_path, multimodal=bool(images), device_map=args.device_map, torch_dtype=args.torch_dtype)
    response = runner.generate(args.prompt, images=images, max_new_tokens=args.max_new_tokens)
    print(response)
    write_outputs(args.output_dir, "quick_start_with_model_garden", vars(args) | {"response": response})


if __name__ == "__main__":
    main()