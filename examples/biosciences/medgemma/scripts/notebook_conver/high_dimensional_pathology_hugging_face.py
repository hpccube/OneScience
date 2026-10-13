#!/usr/bin/env python3
import argparse
import gzip
import io
import random
from pathlib import Path

from medgemma_script_utils import LocalMedGemmaRunner, add_common_args, load_image, write_outputs


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_PROMPT = (
    "Analyze only the pathology patch provided. Describe the visible morphology, give the most likely "
    "interpretation and a short differential diagnosis, and state the confidence and limitations. "
    "Do not assume findings that are not visible in the image."
)


def strip_thinking_component(text: str) -> str:
    cleaned = str(text)
    if "<unused95>" in cleaned:
        cleaned = cleaned.split("<unused95>")[-1]
    cleaned = cleaned.replace("<unused94>", "")
    cleaned = cleaned.replace("<unused95>", "")
    if "thought" in cleaned:
        cleaned = cleaned.split("thought", 1)[0]
    return cleaned.strip()



def _h5_patch_files(root: Path) -> list[Path]:
    preferred = sorted(root.glob("*_x.h5.gz")) + sorted(root.glob("*_x.h5"))
    if preferred:
        return preferred
    return sorted(root.glob("*.h5.gz")) + sorted(root.glob("*.h5"))


def _read_h5_patches(path: Path, limit: int, seed: int) -> list[dict]:
    import h5py
    import numpy as np
    from PIL import Image

    if path.name.endswith(".h5.gz"):
        with gzip.open(path, "rb") as handle:
            payload = io.BytesIO(handle.read())
        h5_file = h5py.File(payload, "r")
    else:
        h5_file = h5py.File(path, "r")

    with h5_file:
        key = "x" if "x" in h5_file else next(iter(h5_file.keys()))
        dataset = h5_file[key]
        count = int(dataset.shape[0])
        if count == 0:
            return []
        sample_count = min(limit, count) if limit > 0 else count
        indices = sorted(random.Random(seed).sample(range(count), k=sample_count)) if sample_count < count else list(range(count))
        patches = []
        for index in indices:
            arr = np.asarray(dataset[index])
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.moveaxis(arr, 0, -1)
            arr = arr.astype("uint8")
            patches.append({
                "image": Image.fromarray(arr).convert("RGB"),
                "source": str(path),
                "dataset_index": index,
            })
        return patches


def load_pathology_patches(image_dir: str | None, h5_path: str | None, limit: int, seed: int) -> list[dict]:
    if h5_path:
        path = Path(h5_path)
        if not path.exists():
            raise FileNotFoundError(f"H5 file not found: {path}")
        print(f"Loading pathology patches from {path}")
        return _read_h5_patches(path, limit, seed)

    if not image_dir:
        raise ValueError('Provide --image_dir or --h5_path.')
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Pathology input not found: {root}. Set --image_dir or --h5_path, or create the default notebook-style dataset path."
        )

    if root.is_file():
        if root.name.endswith(".h5") or root.name.endswith(".h5.gz"):
            print(f"Loading pathology patches from {root}")
            return _read_h5_patches(root, limit, seed)
        return [{"image": load_image(str(root)), "source": str(root), "dataset_index": None}]

    image_paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if image_paths:
        if limit > 0 and len(image_paths) > limit:
            image_paths = random.Random(seed).sample(image_paths, k=limit)
        return [
            {"image": load_image(str(path)), "source": str(path), "dataset_index": None}
            for path in image_paths
        ]

    h5_paths = _h5_patch_files(root)
    if h5_paths:
        print(f"Loading pathology patches from {h5_paths[0]}")
        return _read_h5_patches(h5_paths[0], limit, seed)

    raise FileNotFoundError(
        f"No supported pathology inputs found in {root}. Provide image files or a .h5/.h5.gz patch file."
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze pathology image patches with local Hugging Face MedGemma.")
    add_common_args(parser, multimodal=True)
    parser.set_defaults(max_new_tokens=2000)
    parser.add_argument("--image_dir")
    parser.add_argument("--h5_path")
    parser.add_argument("--max_patches", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tissue_context", default="")
    parser.add_argument("--inference_mode", choices=["per_patch", "aggregate"], default="per_patch")
    args = parser.parse_args()

    patches = load_pathology_patches(args.image_dir, args.h5_path, args.max_patches, args.seed)
    if not patches:
        raise ValueError("No pathology patches were loaded.")

    patch_dir = Path(args.output_dir) / "selected_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for patch_number, patch in enumerate(patches, start=1):
        index_suffix = f"_index_{patch['dataset_index']}" if patch["dataset_index"] is not None else ""
        saved_path = patch_dir / f"patch_{patch_number:03d}{index_suffix}.png"
        patch["image"].save(saved_path)
        patch["saved_path"] = str(saved_path)

    runner = LocalMedGemmaRunner(args.model_path, multimodal=True, device_map=args.device_map, torch_dtype=args.torch_dtype)

    context = f"\nClinical/specimen context: {args.tissue_context}" if args.tissue_context else ""
    prompt = args.prompt + context
    patch_results = []
    if args.inference_mode == "aggregate":
        response = strip_thinking_component(
            runner.generate(prompt, images=[patch["image"] for patch in patches], max_new_tokens=args.max_new_tokens)
        )
        print(response)
    else:
        for patch_number, patch in enumerate(patches, start=1):
            print(f"Analyzing patch {patch_number}/{len(patches)} (dataset index: {patch['dataset_index']})")
            patch_response = strip_thinking_component(
                runner.generate(prompt, images=[patch["image"]], max_new_tokens=args.max_new_tokens)
            )
            print(patch_response)
            patch_results.append({
                "patch_id": patch_number,
                "source": patch["source"],
                "dataset_index": patch["dataset_index"],
                "saved_path": patch["saved_path"],
                "response": patch_response,
            })
        response = "\n\n".join(
            f"## Patch {item['patch_id']} (dataset index: {item['dataset_index']})\n\n{item['response']}"
            for item in patch_results
        )

    result = vars(args) | {
        "num_images": len(patches),
        "response": response,
        "patch_results": patch_results,
    }
    write_outputs(args.output_dir, "high_dimensional_pathology_hugging_face", result)
    md_path = Path(args.output_dir) / "high_dimensional_pathology_hugging_face.md"
    md_path.write_text(response + "\n", encoding="utf-8")
    print(f"Saved markdown result to {md_path}")


if __name__ == "__main__":
    main()