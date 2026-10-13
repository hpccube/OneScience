#!/usr/bin/env python3
import argparse
from pathlib import Path

from medgemma_script_utils import LocalMedGemmaRunner, add_common_args, dicom_to_image, write_outputs, xray_dicom_to_image


def strip_thinking_component(text: str) -> str:
    cleaned = str(text).replace("<unused94>", "").replace("<unused95>", "")
    if "thought" in cleaned:
        cleaned = cleaned.split("thought", 1)[0]
    return cleaned.strip()


def needs_report_revision(text: str) -> bool:
    lowered = str(text).lower()
    clear_claim = any(term in lowered for term in (
        "lungs are clear",
        "lung fields are clear",
        "no focal pulmonary opacity",
        "no focal air-space opacity",
    ))
    abnormal_claim = any(term in lowered for term in (
        "opacity",
        "infiltrate",
        "nodule",
        "atelectasis",
        "consolidation",
        "pleural effusion",
        "pneumothorax",
    ))
    return clear_claim and abnormal_claim

def main():
    parser = argparse.ArgumentParser(description="Quick start MedGemma inference from a local chest X-ray DICOM.")
    add_common_args(parser, multimodal=True)
    parser.add_argument("--dicom_path", required=True)
    parser.add_argument(
        "--revision_pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a second pass when the draft report contains contradictory statements.",
    )
    parser.add_argument("--prompt", default=(
        "You are an expert radiologist. Analyze this chest X-ray. Report only findings visible in the image. "
        "Use the sections FINDINGS and IMPRESSION. Keep the impression consistent with the findings; "
        "do not say the lungs are clear if you report an opacity. When a finding is uncertain, say "
        "indeterminate and do not invent a diagnosis. Before finalizing, check the report for internal contradictions."
    ))
    args = parser.parse_args()

    dicom_path = Path(args.dicom_path)
    if not dicom_path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    import pydicom

    header = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
    modality = str(getattr(header, "Modality", "")).upper()
    radiograph_modalities = {"CR", "DX", "DR", "MG", "RF", "XA"}
    if modality in radiograph_modalities:
        image, dicom_metadata = xray_dicom_to_image(str(dicom_path))
        image_prompt_context = "This is a projection radiograph."
    else:
        image = dicom_to_image(str(dicom_path), notebook_windowing=True)
        dicom_metadata = {
            "modality": modality,
            "body_part_examined": str(getattr(header, "BodyPartExamined", "")),
            "study_description": str(getattr(header, "StudyDescription", "")),
            "series_description": str(getattr(header, "SeriesDescription", "")),
            "photometric_interpretation": str(getattr(header, "PhotometricInterpretation", "")),
            "rows": int(getattr(header, "Rows", image.height)),
            "columns": int(getattr(header, "Columns", image.width)),
        }
        dicom_metadata = {
            key: value for key, value in dicom_metadata.items() if value not in (None, "")
        }
        image_prompt_context = "This is a single CT image slice."

    default_prompt = (
        "You are an expert radiologist. Analyze this chest X-ray. Report only findings visible in the image. "
        "Use the sections FINDINGS and IMPRESSION. Keep the impression consistent with the findings; "
        "do not say the lungs are clear if you report an opacity. When a finding is uncertain, say "
        "indeterminate and do not invent a diagnosis. Before finalizing, check the report for internal contradictions."
    )
    prompt = args.prompt
    if args.prompt == default_prompt and modality not in radiograph_modalities:
        prompt = (
            "You are an expert radiologist. Analyze this chest CT image slice. Report only findings visible "
            "in this slice. Use the sections FINDINGS and IMPRESSION. Keep the impression consistent with "
            "the findings; when a finding is uncertain, say indeterminate and do not invent a diagnosis. "
            "Before finalizing, check the report for internal contradictions."
        )
    prompt = f"{prompt}\n\n{image_prompt_context}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "quick_start_with_dicom_input.png"
    image.save(preview_path)
    print(f"Saved model input preview to {preview_path}")

    runner = LocalMedGemmaRunner(
        args.model_path,
        multimodal=True,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    draft_response = strip_thinking_component(
        runner.generate(prompt, images=[image], max_new_tokens=args.max_new_tokens)
    )
    response = draft_response
    revision_applied = False
    if args.revision_pass and needs_report_revision(draft_response):
        revision_prompt = (
            "Review the draft report below against the attached image. Return a corrected report "
            "using FINDINGS and IMPRESSION. Remove contradictions, distinguish visible findings "
            "from possible explanations, and do not add unsupported findings. If the image is "
            "indeterminate, state that clearly.\n\nDRAFT REPORT:\n"
            f"{draft_response}"
        )
        response = strip_thinking_component(
            runner.generate(revision_prompt, images=[image], max_new_tokens=args.max_new_tokens)
        )
        revision_applied = True
    print(response)

    result = vars(args) | {
        "dicom_metadata": dicom_metadata,
        "input_preview": str(preview_path),
        "prompt_used": prompt,
        "draft_response": draft_response,
        "revision_applied": revision_applied,
        "response": response,
    }
    write_outputs(args.output_dir, "quick_start_with_dicom", result)
    md_path = output_dir / "quick_start_with_dicom.md"
    md_path.write_text(f"# DICOM Inference Result\n\n{response}\n", encoding="utf-8")
    print(f"Saved markdown result to {md_path}")


if __name__ == "__main__":
    main()