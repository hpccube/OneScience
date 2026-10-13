#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional


def _load_json_records(path: Path):
    try:
        if path.suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flatten_resources(payload):
    resources = []
    payloads = payload if isinstance(payload, list) else [payload]
    for item in payloads:
        if not isinstance(item, dict):
            continue
        if item.get("resourceType") == "Bundle":
            for entry in item.get("entry", []):
                resource = entry.get("resource", {}) if isinstance(entry, dict) else {}
                if isinstance(resource, dict):
                    resources.append(resource)
        elif item.get("resourceType"):
            resources.append(item)
    return resources


def _ref_id(reference: str, resource_type: str) -> str:
    prefix = f"{resource_type}/"
    if prefix in reference:
        return reference.split(prefix, 1)[1].split("/", 1)[0]
    return reference.rsplit(":", 1)[-1].rsplit("/", 1)[-1]


def _patient_id(resource: dict) -> Optional[str]:
    if resource.get("resourceType") == "Patient" and resource.get("id"):
        return str(resource["id"])
    for key in ("patient", "subject"):
        value = resource.get(key)
        if isinstance(value, dict) and value.get("reference"):
            return _ref_id(str(value["reference"]), "Patient")
    return None


def _codeable_text(value) -> str:
    if not isinstance(value, dict):
        return ""
    parts = []
    text = str(value.get("text", "")).strip()
    if text:
        parts.append(text)
    for coding in value.get("coding", []):
        if not isinstance(coding, dict):
            continue
        display = str(coding.get("display", "")).strip()
        if display:
            parts.append(display)
        elif not parts:
            code = str(coding.get("code", "")).strip()
            if code:
                parts.append(code)
    unique_parts = list(dict.fromkeys(parts))
    return " ".join(unique_parts)


def _mentions_sepsis(resource: dict) -> bool:
    searchable = [str(resource.get("reason", "")), _codeable_text(resource.get("code"))]
    for key in ("reasonCode", "reasonReference", "diagnosis"):
        values = resource.get(key, [])
        if isinstance(values, dict):
            values = [values]
        for value in values:
            if not isinstance(value, dict):
                continue
            searchable.append(str(value.get("display", "")))
            searchable.append(_codeable_text(value))
            condition = value.get("condition")
            if isinstance(condition, dict):
                searchable.append(str(condition.get("display", "")))
                searchable.append(str(condition.get("reference", "")))
    return any(re.search(r"\bsepsis\b|\bseptic\b", item, flags=re.IGNORECASE) for item in searchable)


def _encounter_id(resource: dict) -> Optional[str]:
    context = resource.get("context") or resource.get("encounter")
    if isinstance(context, dict) and context.get("reference"):
        return _ref_id(str(context["reference"]), "Encounter")
    return None


def _medication_text(resource: dict) -> str:
    return _codeable_text(resource.get("medicationCodeableConcept")) or "Unknown medication"


def _find_matches(resources: list[dict]):
    sepsis_conditions = {
        str(resource.get("id"))
        for resource in resources
        if resource.get("resourceType") == "Condition" and resource.get("id") and _mentions_sepsis(resource)
    }
    sepsis_encounters = {
        _encounter_id(resource)
        for resource in resources
        if resource.get("resourceType") == "Condition" and _mentions_sepsis(resource)
    }
    sepsis_encounters.discard(None)

    for encounter in resources:
        if encounter.get("resourceType") != "Encounter":
            continue
        for diagnosis in encounter.get("diagnosis", []):
            if not isinstance(diagnosis, dict):
                continue
            condition = diagnosis.get("condition")
            if isinstance(condition, dict):
                condition_ref = str(condition.get("reference", ""))
                condition_display = str(condition.get("display", ""))
                if _ref_id(condition_ref, "Condition") in sepsis_conditions or re.search(r"\bsepsis\b|\bseptic\b", condition_display, flags=re.IGNORECASE):
                    if encounter.get("id"):
                        sepsis_encounters.add(str(encounter["id"]))

    matches = []
    for resource in resources:
        if resource.get("resourceType") != "MedicationAdministration":
            continue
        reason_condition_ids = set()
        for reason in resource.get("reasonReference", []):
            if isinstance(reason, dict) and reason.get("reference"):
                reason_condition_ids.add(_ref_id(str(reason["reference"]), "Condition"))
        if _mentions_sepsis(resource) or bool(reason_condition_ids & sepsis_conditions) or (_encounter_id(resource) in sepsis_encounters):
            matches.append(resource)
    return matches


def _json_paths(data_dir: Path):
    return sorted(list(data_dir.rglob("*.json")) + list(data_dir.rglob("*.jsonl")))


def scan(data_dir: Path):
    for path in _json_paths(data_dir):
        resources = _flatten_resources(_load_json_records(path))
        if not resources:
            continue
        patient_ids = sorted({pid for resource in resources if (pid := _patient_id(resource))})
        matches = _find_matches(resources)
        if matches and patient_ids:
            yield path, patient_ids[0], matches


def scan_any_medication_patient(data_dir: Path):
    for path in _json_paths(data_dir):
        resources = _flatten_resources(_load_json_records(path))
        if not resources:
            continue
        patient_ids = sorted({pid for resource in resources if (pid := _patient_id(resource))})
        matches = [resource for resource in resources if resource.get("resourceType") == "MedicationAdministration"]
        if matches and patient_ids:
            yield path, patient_ids[0], matches


def main():
    parser = argparse.ArgumentParser(description="Find a FHIR patient with sepsis-related MedicationAdministration records.")
    parser.add_argument("--fhir_data_dir", required=True)
    parser.add_argument("--first", action="store_true", help="Print only the first matching patient ID.")
    parser.add_argument("--fallback-any-medication", action="store_true", help="If no sepsis-related medication patient is found, return a patient with any MedicationAdministration records.")
    args = parser.parse_args()

    data_dir = Path(args.fhir_data_dir)
    candidates = list(scan(data_dir))
    used_fallback = False
    if not candidates and args.fallback_any_medication:
        candidates = list(scan_any_medication_patient(data_dir))
        used_fallback = True

    for path, patient_id, matches in candidates:
        if args.first:
            print(patient_id)
            return
        meds = "; ".join(
            f"{_medication_text(resource)} at {resource.get('effectiveDateTime', 'unknown time')}"
            for resource in matches[:5]
        )
        label = "fallback-any-medication" if used_fallback else "sepsis-related-medication"
        print(f"{patient_id}\t{label}\t{path}\t{meds}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()