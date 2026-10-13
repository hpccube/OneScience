#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Iterable

from PIL import Image


def default_model_path(multimodal: bool = True) -> str:
    datasets_dir = os.getenv("ONESCIENCE_DATASETS_DIR", "")
    if multimodal:
        return str(Path(datasets_dir) / "medgemma/modelscope/google/medgemma-1.5-4b-it")
    return str(Path(datasets_dir) / "medgemma/model_garden/google--medgemma-27b-text-it/snapshots/master")


def load_image(path: str) -> Image.Image:
    if path.startswith("http://") or path.startswith("https://"):
        import requests

        response = requests.get(path, timeout=60)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(path).convert("RGB")


def load_images_from_dir(image_dir: str, limit: int = 8) -> list[Image.Image]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in sorted(Path(image_dir).iterdir()) if p.suffix.lower() in exts]
    return [load_image(str(path)) for path in paths[:limit]]


def _dicom_paths(dicom_dir: str) -> list[Path]:
    root = Path(dicom_dir)
    paths = sorted(root.rglob("*.dcm"))
    if not paths:
        paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
    return paths


def _dicom_sort_key(path: Path) -> tuple[int, float, str]:
    import pydicom

    try:
        dcm = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return (1, 0.0, str(path))
    if getattr(dcm, "InstanceNumber", None) is not None:
        try:
            return (0, float(dcm.InstanceNumber), str(path))
        except Exception:
            pass
    if getattr(dcm, "ImagePositionPatient", None) is not None:
        try:
            return (0, float(dcm.ImagePositionPatient[-1]), str(path))
        except Exception:
            pass
    return (1, 0.0, str(path))


def filter_dicom_paths(
    dicom_dir: str,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
) -> list[Path]:
    import pydicom

    paths = _dicom_paths(dicom_dir)
    if not study_instance_uid and not series_instance_uid:
        return sorted(paths, key=_dicom_sort_key)

    matched = []
    for path in paths:
        try:
            dcm = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if study_instance_uid and str(getattr(dcm, "StudyInstanceUID", "")) != study_instance_uid:
            continue
        if series_instance_uid and str(getattr(dcm, "SeriesInstanceUID", "")) != series_instance_uid:
            continue
        matched.append(path)
    return sorted(matched, key=_dicom_sort_key)


def sample_dicom_paths(paths: list[Path], max_slices: int = 85) -> list[Path]:
    if max_slices <= 0 or len(paths) <= max_slices:
        return paths
    # Match the original notebook: skip the first exact index and sample uniformly through the volume.
    return [paths[int(round(i / max_slices * (len(paths) - 1)))] for i in range(1, max_slices + 1)]


def _norm_window(arr, lo: float, hi: float):
    import numpy as np

    arr = np.clip(arr, lo, hi).astype("float32")
    arr -= lo
    arr /= hi - lo
    arr *= 255.0
    return arr


def dicom_to_image(path: str, notebook_windowing: bool = True) -> Image.Image:
    import numpy as np
    import pydicom

    dcm = pydicom.dcmread(path)
    arr = pydicom.pixels.apply_rescale(dcm.pixel_array, dcm).astype("float32")
    if notebook_windowing:
        window_clips = [(-1024.0, 1024.0), (-135.0, 215.0), (0.0, 80.0)]
        rgb = np.stack([_norm_window(arr, lo, hi) for lo, hi in window_clips], axis=-1)
        rgb = np.round(rgb, 0).astype("uint8")
        return Image.fromarray(rgb).convert("RGB")

    lo, hi = -160.0, 240.0
    arr = _norm_window(arr, lo, hi).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def xray_dicom_to_image(path: str) -> tuple[Image.Image, dict]:
    """Convert a radiograph DICOM to display-ready RGB using its DICOM LUT metadata."""
    import numpy as np
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array
    if arr.ndim > 2 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

    if arr.ndim == 2:
        arr = apply_modality_lut(arr, dcm)
        try:
            arr = apply_voi_lut(arr, dcm)
        except Exception:
            pass
        arr = np.asarray(arr, dtype="float32")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError(f"DICOM pixel data contains no finite values: {path}")
        lo, hi = np.percentile(finite, (0.5, 99.5))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            gray = np.zeros(arr.shape, dtype="uint8")
        else:
            gray = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                gray = 1.0 - gray
            gray = np.round(gray * 255.0).astype("uint8")
        image = Image.fromarray(gray, mode="L").convert("RGB")
    else:
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = arr.astype("float32")
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            arr = (
                np.zeros(arr.shape, dtype="uint8")
                if hi <= lo
                else np.round((arr - lo) / (hi - lo) * 255.0).astype("uint8")
            )
        image = Image.fromarray(arr[..., :3]).convert("RGB")

    metadata = {
        "modality": str(getattr(dcm, "Modality", "")),
        "view_position": str(getattr(dcm, "ViewPosition", "")),
        "body_part_examined": str(getattr(dcm, "BodyPartExamined", "")),
        "study_description": str(getattr(dcm, "StudyDescription", "")),
        "series_description": str(getattr(dcm, "SeriesDescription", "")),
        "photometric_interpretation": str(getattr(dcm, "PhotometricInterpretation", "")),
        "rows": int(getattr(dcm, "Rows", image.height)),
        "columns": int(getattr(dcm, "Columns", image.width)),
    }
    return image, {key: value for key, value in metadata.items() if value not in (None, "")}

def ct_dicom_dir_to_images(
    dicom_dir: str,
    max_slices: int = 85,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    notebook_windowing: bool = True,
) -> list[Image.Image]:
    paths = filter_dicom_paths(dicom_dir, study_instance_uid, series_instance_uid)
    if not paths and (study_instance_uid or series_instance_uid):
        print("No DICOM files matched the requested Study/Series UID; falling back to all files in the directory.")
        paths = filter_dicom_paths(dicom_dir)
    paths = sample_dicom_paths(paths, max_slices)
    return [dicom_to_image(str(path), notebook_windowing=notebook_windowing) for path in paths]


def dicom_dir_metadata(dicom_dir: str, study_instance_uid: str | None = None, series_instance_uid: str | None = None) -> dict:
    """Read non-pixel DICOM identifiers from the first file in a CT series."""
    import pydicom

    root = Path(dicom_dir)
    all_paths = _dicom_paths(dicom_dir)
    paths = filter_dicom_paths(dicom_dir, study_instance_uid, series_instance_uid)
    if not paths and (study_instance_uid or series_instance_uid):
        paths = filter_dicom_paths(dicom_dir)
    metadata = {
        "source_type": "dicom_dir",
        "dicom_dir": str(root.resolve()),
        "dicom_file_count": len(paths),
        "dicom_total_file_count": len(all_paths),
        "requested_study_instance_uid": study_instance_uid,
        "requested_series_instance_uid": series_instance_uid,
        "sample_name": root.name,
    }
    if not paths:
        return metadata

    metadata["first_dicom_file"] = str(paths[0])
    try:
        dcm = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
    except Exception as exc:
        metadata["metadata_error"] = str(exc)
        return metadata

    fields = {
        "patient_name": "PatientName",
        "patient_id": "PatientID",
        "patient_sex": "PatientSex",
        "patient_birth_date": "PatientBirthDate",
        "study_date": "StudyDate",
        "study_description": "StudyDescription",
        "series_description": "SeriesDescription",
        "study_instance_uid": "StudyInstanceUID",
        "series_instance_uid": "SeriesInstanceUID",
    }
    for key, attr in fields.items():
        value = getattr(dcm, attr, None)
        if value not in (None, ""):
            metadata[key] = str(value)
    return metadata


def image_dir_metadata(image_dir: str) -> dict:
    root = Path(image_dir)
    return {
        "source_type": "image_dir",
        "image_dir": str(root.resolve()),
        "sample_name": root.name,
    }


def format_source_metadata(metadata: dict) -> str:
    if not metadata:
        return "Source: unknown"
    lines = ["Source metadata:"]
    ordered = [
        ("sample_name", "Sample"),
        ("patient_id", "Patient ID"),
        ("patient_name", "Patient Name"),
        ("patient_sex", "Patient Sex"),
        ("patient_birth_date", "Patient Birth Date"),
        ("study_date", "Study Date"),
        ("study_description", "Study Description"),
        ("series_description", "Series Description"),
        ("dicom_file_count", "Matched DICOM Files"),
        ("dicom_total_file_count", "Total DICOM Files"),
        ("requested_study_instance_uid", "Requested Study Instance UID"),
        ("requested_series_instance_uid", "Requested Series Instance UID"),
        ("dicom_dir", "DICOM Dir"),
        ("image_dir", "Image Dir"),
        ("study_instance_uid", "Study Instance UID"),
        ("series_instance_uid", "Series Instance UID"),
    ]
    for key, label in ordered:
        if metadata.get(key) not in (None, ""):
            lines.append(f"{label}: {metadata[key]}")
    return "\n".join(lines)

def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class LocalMedGemmaRunner:
    def __init__(self, model_path: str, multimodal: bool = True, device_map: str = "auto", torch_dtype: str = "auto"):
        import torch

        self.torch = torch
        dtype = torch_dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32

        self.multimodal = multimodal
        if multimodal:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=dtype,
                local_files_only=True,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.processor = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=dtype,
                local_files_only=True,
            )

    def generate_messages(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        if not self.multimodal:
            raise ValueError("generate_messages is only available for multimodal models.")
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def generate(self, prompt: str, images: Iterable[Image.Image] | None = None, max_new_tokens: int = 512) -> str:
        images = list(images or [])
        if self.multimodal:
            content = [{"type": "text", "text": prompt}]
            content.extend({"type": "image", "image": image} for image in images)
            messages = [{"role": "user", "content": content}]
            return self.generate_messages(messages, max_new_tokens=max_new_tokens)

        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


def run_vertex_prediction(project_id: str, region: str, endpoint_id: str, prompt: str, images: list[Image.Image], max_tokens: int):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_to_data_url(image)}} for image in images)
    request = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    return endpoint.raw_predict(body=json.dumps(request), headers={"Content-Type": "application/json"}).data.decode()


def add_common_args(parser: argparse.ArgumentParser, multimodal: bool = True):
    parser.add_argument("--model_path", default=os.getenv("MEDGEMMA_MODEL_PATH", default_model_path(multimodal)))
    parser.add_argument("--device_map", default=os.getenv("MEDGEMMA_DEVICE_MAP", "auto"))
    parser.add_argument("--torch_dtype", default=os.getenv("MEDGEMMA_TORCH_DTYPE", "auto"))
    parser.add_argument("--max_new_tokens", type=int, default=int(os.getenv("MAX_NEW_TOKENS", "512")))
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "./outputs"))


def write_outputs(output_dir: str, name: str, result: dict):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / f"{name}.json"
    txt_path = path / f"{name}.txt"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    text = str(result.get("response", result))
    if result.get("source_metadata"):
        text = format_source_metadata(result["source_metadata"]) + "\n\n" + text
    txt_path.write_text(text, encoding="utf-8")
    print(f"Saved JSON result to {json_path}")
    print(f"Saved text result to {txt_path}")
