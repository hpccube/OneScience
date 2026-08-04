#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from medgemma_script_utils import (
    LocalMedGemmaRunner,
    add_common_args,
    ct_dicom_dir_to_images,
    dicom_dir_metadata,
    format_source_metadata,
    image_dir_metadata,
    load_images_from_dir,
    write_outputs,
)

NOTEBOOK_STUDY_INSTANCE_UID = "1.3.6.1.4.1.14519.5.2.1.9203.8273.982856921320609617394372605436"
NOTEBOOK_SERIES_INSTANCE_UID = "1.3.6.1.4.1.14519.5.2.1.9203.8273.275179554444442893192427753220"
NOTEBOOK_INSTRUCTION = (
    "You are an instructor teaching medical students. You are analyzing a contiguous block "
    "of CT slices from the center of the abdomen. Please review the slices provided below carefully."
)
NOTEBOOK_QUERY = (
    "\n\nBased on the visual evidence in the slices provided above, is this image a good teaching example "
    "of liver pathology? Comment on hypodense lesions or other hepatic irregularities. Do not comment "
    "on findings outside the liver. Please provide your reasoning and conclude with a 'Final Answer: yes' "
    "or 'Final Answer: no'."
)


def default_ct_dicom_dir() -> str | None:
    datasets_dir = os.getenv("ONESCIENCE_DATASETS_DIR", "")
    candidates = [
        Path(datasets_dir) / "medgemma" / "CTLM",
        Path(datasets_dir) / "medgemma" / "ct_dicom",
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return str(candidate)
    return str(candidates[0]) if datasets_dir else None


def build_notebook_messages(instruction: str, query: str, images) -> list[dict]:
    content = [{"type": "text", "text": instruction}]
    for slice_number, image in enumerate(images, 1):
        content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": f"SLICE {slice_number}"})
    content.append({"type": "text", "text": query})
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description="Analyze CT slices with local Hugging Face MedGemma.")
    add_common_args(parser, multimodal=True)
    parser.set_defaults(max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "2000")))
    parser.add_argument("--dicom_dir", default=os.getenv("CT_DICOM_DIR") or default_ct_dicom_dir())
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--max_slices", type=int, default=int(os.getenv("CT_MAX_SLICES", "85")))
    parser.add_argument("--study_instance_uid", default=os.getenv("CT_STUDY_INSTANCE_UID", NOTEBOOK_STUDY_INSTANCE_UID))
    parser.add_argument("--series_instance_uid", default=os.getenv("CT_SERIES_INSTANCE_UID", NOTEBOOK_SERIES_INSTANCE_UID))
    parser.add_argument("--instruction", default=os.getenv("CT_INSTRUCTION") or NOTEBOOK_INSTRUCTION)
    parser.add_argument("--prompt", default=os.getenv("CT_PROMPT") or NOTEBOOK_QUERY)
    parser.add_argument("--simple_windowing", action="store_true", help="Use the older single-window CT preprocessing instead of notebook RGB windowing.")
    args = parser.parse_args()

    if args.dicom_dir and args.image_dir:
        raise ValueError("Provide only one of --dicom_dir or --image_dir.")

    if args.dicom_dir:
        source_metadata = dicom_dir_metadata(args.dicom_dir, args.study_instance_uid, args.series_instance_uid)
        images = ct_dicom_dir_to_images(
            args.dicom_dir,
            args.max_slices,
            study_instance_uid=args.study_instance_uid,
            series_instance_uid=args.series_instance_uid,
            notebook_windowing=not args.simple_windowing,
        )
    elif args.image_dir:
        source_metadata = image_dir_metadata(args.image_dir)
        images = load_images_from_dir(args.image_dir, args.max_slices)
    else:
        raise ValueError("Provide --dicom_dir or --image_dir.")

    print(format_source_metadata(source_metadata))
    print(f"Prompt slices: {len(images)}")
    print(f"Requested max slices: {args.max_slices}")
    print(f"Notebook-style CT windowing: {not args.simple_windowing}")
    print(f"Notebook default prompt: {args.instruction == NOTEBOOK_INSTRUCTION and args.prompt == NOTEBOOK_QUERY}")
    print(f"Max new tokens: {args.max_new_tokens}")

    runner = LocalMedGemmaRunner(args.model_path, multimodal=True, device_map=args.device_map, torch_dtype=args.torch_dtype)
    messages = build_notebook_messages(args.instruction, args.prompt, images)
    response = runner.generate_messages(messages, max_new_tokens=args.max_new_tokens)
    print(response)
    write_outputs(
        args.output_dir,
        "high_dimensional_ct_hugging_face",
        vars(args) | {
            "num_images": len(images),
            "notebook_study_instance_uid": NOTEBOOK_STUDY_INSTANCE_UID,
            "notebook_series_instance_uid": NOTEBOOK_SERIES_INSTANCE_UID,
            "source_metadata": source_metadata,
            "response": response,
        },
    )


if __name__ == "__main__":
    main()