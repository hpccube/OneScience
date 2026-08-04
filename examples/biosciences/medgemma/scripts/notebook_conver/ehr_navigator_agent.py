#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Annotated, Optional, TypedDict
from urllib.parse import quote
import operator
import sys

import torch
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import render_text_description, tool
from langgraph.graph import END, StateGraph


class LocalMedGemmaLLM:
    def __init__(self, model_path: str, device_map: str = "auto", torch_dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = None
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if hasattr(self.model, 'hf_device_map'):
            print(f"[INFO] Model device map: {self.model.hf_device_map}")

    def invoke(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0):
        try:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt

            if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print(f"[DEBUG] Corrected pad_token_id to {self.tokenizer.pad_token_id}")

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            do_sample = temperature is not None and temperature > 0
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            print(f"[DEBUG] Generated {len(new_tokens)} new tokens (first 20 IDs: {new_tokens[:20].tolist()})")
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return decoded
        except Exception as e:
            print(f"[ERROR] Invoke failed: {e}")
            import traceback
            traceback.print_exc()
            return ""


DEMO_SEPSIS_PATIENT_ID = "e4350e97-bb8c-70b7-9997-9e098cfacef8"

FHIR_RESOURCE_TYPES = [
    "Encounter", "Practitioner", "Condition", "Observation", "AllergyIntolerance",
    "FamilyMemberHistory", "MedicationRequest", "MedicationStatement",
    "MedicationAdministration", "DiagnosticReport", "Procedure", "ServiceRequest",
]


def strip_json_decoration(text: str) -> str:
    cleaned = str(text).strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        return cleaned[7:-3].strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        return cleaned[3:-3].strip()
    return cleaned


def exclude_thinking_component(text: str) -> str:
    """Remove thinking blocks and thought-prefixed lines from LLM output."""
    cleaned = str(text)

    # Remove standard MedGemma thinking traces.
    cleaned = re.sub(r"<unused94>.*?<unused95>", "", cleaned, flags=re.DOTALL)

    # Some decoders drop <unused94> but keep the leading "thought" text and
    # <unused95>. Strip those leading bare thought blocks before the answer.
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r"(?is)^\s*thought\b.*?<unused95>\s*", "", cleaned)

    # If generation continues with a new thought section after the answer, keep
    # the answer and discard the repeated reasoning tail.
    parts = re.split(r"(?im)^\s*thought\b", cleaned, maxsplit=1)
    if parts[0].strip():
        cleaned = parts[0]

    cleaned = cleaned.replace("<unused94>", "").replace("<unused95>", "")
    cleaned = re.sub(r"(?im)^\s*thought\b[^\n]*(\n|$)", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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


def _reference_id(reference: str, resource_type: str) -> str:
    prefix = f"{resource_type}/"
    if prefix in reference:
        return reference.split(prefix, 1)[1].split("/", 1)[0]
    return reference


def _resource_ref(resource: dict) -> str:
    resource_type = resource.get("resourceType", "Resource")
    resource_id = resource.get("id")
    return f"{resource_type}/{resource_id}" if resource_id else resource_type

def _patient_id_from_question(question: str) -> Optional[str]:
    match = re.search(r"Patient ID\s+([A-Za-z0-9_.-]+)", question)
    return match.group(1).rstrip(".") if match else None


def _is_sepsis_medication_question(question: str) -> bool:
    question_lower = question.lower()
    medication_terms = ["medication", "medications", "administered", "drug", "drugs"]
    return "sepsis" in question_lower and any(term in question_lower for term in medication_terms)


def _resource_encounter_id(resource: dict) -> Optional[str]:
    context = resource.get("context") or resource.get("encounter")
    if isinstance(context, dict):
        reference = str(context.get("reference", ""))
        if reference:
            return _reference_id(reference, "Encounter")
    return None


def _resource_mentions_sepsis(resource: dict) -> bool:
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


def _summarize_sepsis_medication_administrations(tool_output: str) -> Optional[str]:
    try:
        payload = json.loads(tool_output)
    except json.JSONDecodeError:
        return None

    entries = payload.get("entry", []) if isinstance(payload, dict) else []
    resources = [entry.get("resource", {}) for entry in entries if isinstance(entry, dict)]
    sepsis_conditions = {
        resource.get("id")
        for resource in resources
        if resource.get("resourceType") == "Condition" and _resource_mentions_sepsis(resource)
    }
    sepsis_conditions.discard(None)
    sepsis_encounters = {
        _resource_encounter_id(resource)
        for resource in resources
        if resource.get("resourceType") in {"Encounter", "Condition"} and _resource_mentions_sepsis(resource)
    }
    sepsis_encounters.discard(None)

    medication_resources = [
        resource for resource in resources
        if resource.get("resourceType") == "MedicationAdministration"
    ]
    matched = []
    for resource in medication_resources:
        reason_condition_ids = set()
        for reason in resource.get("reasonReference", []):
            if isinstance(reason, dict):
                reference = str(reason.get("reference", ""))
                if reference:
                    reason_condition_ids.add(_reference_id(reference, "Condition"))
        if (
            _resource_mentions_sepsis(resource)
            or bool(reason_condition_ids & sepsis_conditions)
            or (_resource_encounter_id(resource) and _resource_encounter_id(resource) in sepsis_encounters)
        ):
            matched.append(resource)
    if not matched:
        if not medication_resources:
            return "No MedicationAdministration records were found for this patient in the configured FHIR data source."
        if not sepsis_conditions and not sepsis_encounters:
            return "MedicationAdministration records exist for this patient, but no sepsis/septic Condition or Encounter was found in the configured FHIR data source."
        return "No sepsis-related MedicationAdministration records were found for this patient. Check that the patient ID has a sepsis encounter with medication administrations."

    lines = ["Medication administrations relevant to sepsis:"]
    for resource in sorted(matched, key=lambda item: str(item.get("effectiveDateTime", ""))):
        medication = _codeable_text(resource.get("medicationCodeableConcept")) or "Unknown medication"
        effective_period = resource.get("effectivePeriod") if isinstance(resource.get("effectivePeriod"), dict) else {}
        when = resource.get("effectiveDateTime") or effective_period.get("start") or "unknown time"
        lines.append(f"- {medication} at {when} [{_resource_ref(resource)}]")
    return "\n".join(lines)


def _combine_fhir_bundles(tool_outputs: list[str]) -> str:
    entries = []
    for tool_output in tool_outputs:
        try:
            payload = json.loads(tool_output)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.extend(payload.get("entry", []))
    return json.dumps({"resourceType": "Bundle", "type": "searchset", "total": len(entries), "entry": entries})

def _extract_balanced_json(text: str, opener: str, closer: str):
    cleaned = str(text).strip()
    start = cleaned.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        char = cleaned[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_json_object(text: str) -> Optional[dict]:
    value = _extract_balanced_json(text, "{", "}")
    return value if isinstance(value, dict) else None


def safe_extract_json(text: str):
    """Extract a JSON object or list from LLM output."""
    cleaned = str(text).replace("```json", "").replace("```", "")
    first_object = cleaned.find("{")
    first_array = cleaned.find("[")
    if first_object != -1 and (first_array == -1 or first_object < first_array):
        value = _extract_balanced_json(cleaned, "{", "}")
    else:
        value = _extract_balanced_json(cleaned, "[", "]")
    if value is not None:
        return value

    cleaned = exclude_thinking_component(cleaned)
    for opener, closer in (("{", "}"), ("[", "]")):
        value = _extract_balanced_json(cleaned, opener, closer)
        if value is not None:
            return value
    return None


def llm_text(response) -> str:
    return response.content if hasattr(response, "content") else str(response)


def _format_manifest_codes(codes: list[str], limit: int = 80) -> str:
    unique_codes = list(dict.fromkeys(codes))
    rendered = ", ".join(unique_codes[:limit])
    if len(unique_codes) > limit:
        rendered += f", ... ({len(unique_codes) - limit} more omitted)"
    return rendered


def _load_json_file(path: Path):
    if path.suffix == ".jsonl":
        entries = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
    return json.loads(path.read_text(encoding="utf-8"))


def _resource_matches_patient(resource: dict, patient_id: str) -> bool:
    references = []
    for key in ("patient", "subject"):
        value = resource.get(key)
        if isinstance(value, dict):
            references.append(value.get("reference", ""))
    if not references:
        return False
    return any(ref.endswith(f"Patient/{patient_id}") or ref.endswith(patient_id) for ref in references)


def _resource_matches_filter(resource: dict, filter_code: Optional[str]) -> bool:
    if not filter_code:
        return True
    requested = {item.strip().lower() for item in filter_code.split(",") if item.strip()}
    if not requested:
        return True
    candidates = []
    for field_name in ("code", "category"):
        field = resource.get(field_name)
        fields = field if isinstance(field, list) else [field]
        for item in fields:
            if not isinstance(item, dict):
                continue
            candidates.append(str(item.get("text", "")).lower())
            for coding in item.get("coding", []):
                candidates.append(str(coding.get("code", "")).lower())
                candidates.append(str(coding.get("display", "")).lower())
    return any(req in candidate or candidate in req for req in requested for candidate in candidates if candidate)


def _extract_resources(payload, resource_type: str, patient_id: str, filter_code: Optional[str]) -> list[dict]:
    resources = []
    payloads = payload if isinstance(payload, list) else [payload]
    for item in payloads:
        if not isinstance(item, dict):
            continue
        if item.get("resourceType") == "Bundle":
            for entry in item.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == resource_type:
                    resources.append(resource)
        elif item.get("resourceType") == resource_type:
            resources.append(item)
    return [
        resource
        for resource in resources
        if _resource_matches_patient(resource, patient_id) and _resource_matches_filter(resource, filter_code)
    ]


def _load_local_fhir_bundle(data_dir: Path, patient_id: str, resource_type: str, filter_code: Optional[str]) -> dict:
    candidate_files = []
    for base in (data_dir / patient_id, data_dir):
        candidate_files.extend([
            base / f"{resource_type}.json",
            base / f"{resource_type}.jsonl",
            base / f"{resource_type.lower()}.json",
            base / f"{resource_type.lower()}.jsonl",
        ])
    resources = []
    for candidate in candidate_files:
        if candidate.exists():
            resources.extend(_extract_resources(_load_json_file(candidate), resource_type, patient_id, filter_code))
    if not resources:
        for candidate in data_dir.rglob("*.json"):
            resources.extend(_extract_resources(_load_json_file(candidate), resource_type, patient_id, filter_code))
        for candidate in data_dir.rglob("*.jsonl"):
            resources.extend(_extract_resources(_load_json_file(candidate), resource_type, patient_id, filter_code))
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(resources),
        "entry": [{"resource": resource} for resource in resources],
    }


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    patient_fhir_manifest: dict
    tool_output_summary: Annotated[list, operator.add]
    tool_calls_to_execute: list
    relevant_resource_types: list


def build_agent(llm, fhir_store_url: str, local_fhir_data_dir: Optional[str] = None, verbose: bool = False):
    local_fhir_root = Path(local_fhir_data_dir).resolve() if local_fhir_data_dir else None

    def log(message: str) -> None:
        if verbose:
            print(message)

    def _get_fhir_resource(resource_path: str, patient_id: str, resource_type: str, filter_code: Optional[str]) -> dict:
        if local_fhir_root:
            log(f"[FHIR] Loading local {resource_type} resources from {local_fhir_root}")
            return _load_local_fhir_bundle(local_fhir_root, patient_id, resource_type, filter_code)
        try:
            credentials, _ = get_auth_default()
            request = google_auth_requests.Request()
            credentials.refresh(request)
            headers = {"Authorization": f"Bearer {credentials.token}"}
            all_entries = []
            next_url = f"{fhir_store_url.rstrip('/')}/{resource_path}"
            while next_url:
                log(f"[FHIR] GET {next_url[next_url.find('/fhir/'):]}")
                response = requests.get(next_url, headers=headers, timeout=120)
                response.raise_for_status()
                current_page = response.json()
                if "entry" in current_page:
                    all_entries.extend(current_page["entry"])
                next_url = None
                for link in current_page.get("link", []):
                    if link.get("relation") == "next":
                        next_url = link.get("url")
                        break
            full_bundle = {"resourceType": "Bundle", "type": "searchset", "total": len(all_entries), "entry": all_entries}
            def clean(obj):
                if isinstance(obj, list): return [clean(i) for i in obj]
                if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items() if k != "meta"}
                if isinstance(obj, str) and "/fhir/" in obj: return obj.split("/fhir/")[-1]
                return obj
            for entry in all_entries:
                entry.pop("fullUrl", None)
                entry.pop("search", None)
                if "resource" in entry: entry["resource"] = clean(entry["resource"])
            return full_bundle
        except Exception as exc:
            return {"error": f"An error occurred: {exc}"}

    @tool
    def get_patient_fhir_resource(patient_id: str, fhir_resource: str, filter_code: Optional[str] = None) -> str:
        """Retrieve FHIR resources for a patient, optionally filtered by code or category."""
        resource_path = f"{fhir_resource}?patient=Patient/{patient_id}"
        if filter_code: resource_path += f"&code={quote(filter_code.replace(' ', ''))}"
        if "Medication" in fhir_resource: resource_path += f"&_include={fhir_resource}:medication"
        content = _get_fhir_resource(resource_path, patient_id, fhir_resource, filter_code)
        if content.get("total", 0) == 0 and filter_code:
            log("[FHIR] No code results. Retrying with category filter.")
            resource_path = f"{fhir_resource}?patient=Patient/{patient_id}&category={quote(filter_code)}"
            content = _get_fhir_resource(resource_path, patient_id, fhir_resource, filter_code)
        return json.dumps(content)

    @tool
    def get_patient_data_manifest(patient_id: str) -> str:
        """List available FHIR resource codes and displays for a patient."""
        manifest = {}
        for resource_type in FHIR_RESOURCE_TYPES:
            resource_path = f"{resource_type}?patient=Patient/{patient_id}"
            log(f"[FHIR] Discovering {resource_type} resources for patient {patient_id}")
            resources_json = _get_fhir_resource(resource_path, patient_id, resource_type, None)
            if isinstance(resources_json, dict) and resources_json.get("total", 0) > 0:
                for entry in resources_json.get("entry", []):
                    resource = entry.get("resource", {})
                    manifest.setdefault(resource_type, [])
                    if "code" in resource and "coding" in resource["code"]:
                        for code in resource.get("code", {}).get("coding", []):
                            manifest[resource_type].append(f'{code.get("display", "")}={code.get("code", "")}')
        return json.dumps(manifest)

    def call_manifest_tool_node(state):
        last_message = state["messages"][-1]
        extraction_prompt = (
            f"USER QUESTION: {last_message.content}\n\n"
            "You are an API request generator. Your task is to identify the patient ID "
            "from the user's question and output a JSON object to call the get_patient_data_manifest tool.\n\n"
            f"Your available tool is:\n{render_text_description([get_patient_data_manifest])}\n\n"
            "Generate the correct JSON to call the tool. Respond with only a single, raw JSON object.\n\n"
            "EXAMPLE:\n"
            "{\n"
            '  "name": "get_patient_data_manifest",\n'
            '  "args": {\n'
            '    "patient_id": "some-patient-id-from-the-question"\n'
            "  }\n"
            "}\n"
        )
        response_str = llm_text(llm.invoke(extraction_prompt, max_tokens=1000, temperature=0.1))
        tool_call_json = safe_extract_json(response_str)
        if not (tool_call_json and isinstance(tool_call_json, dict) and "args" in tool_call_json):
            patient_match = re.search(r"Patient ID\s+([A-Za-z0-9_.-]+)", last_message.content)
            if not patient_match:
                print(f"[WARN] Could not extract manifest tool call: {response_str[:200]}")
                return {"patient_fhir_manifest": {}}
            tool_call_json = {"args": {"patient_id": patient_match.group(1).rstrip(".")}}
        tool_call_json["args"]["patient_id"] = str(tool_call_json["args"].get("patient_id", "")).rstrip(".")
        try:
            manifest_json = get_patient_data_manifest.invoke(tool_call_json["args"])
            manifest_dict = json.loads(manifest_json)
            print(f"[INFO] Loaded manifest resource types: {', '.join(manifest_dict.keys())}")
            return {"patient_fhir_manifest": manifest_dict}
        except Exception as exc:
            print(f"[WARN] Error calling manifest tool: {exc}")
            return {"patient_fhir_manifest": {}}

    def identify_relevant_resource_types(state):
        manifest = state.get("patient_fhir_manifest", {})
        user_question = state["messages"][1].content
        manifest_content = ""
        for resource_type, codes in manifest.items():
            manifest_content += f"**{resource_type}**: "
            if codes:
                manifest_content += f"Available codes include: {_format_manifest_codes(codes)}\n"
            else:
                manifest_content += "Present (no specific codes found)\n"
        prompt = (
            "SYSTEM INSTRUCTION: think silently if needed.\n"
            f"USER QUESTION: {user_question}\n\n"
            f"PATIENT DATA MANIFEST:\n{manifest_content}\n\n"
            "You are a medical assistant analyzing a patient's FHIR data manifest to answer a user question.\n"
            "Based on the user question, identify the specific FHIR resource types from the manifest that are most likely to contain the information needed to answer the question.\n"
            "Output a JSON list of the relevant resource types. Do not include any other text or formatting.\n"
            "Example:\n[\"Condition\", \"Observation\", \"MedicationRequest\"]\n"
        )
        response_str = llm_text(llm.invoke(prompt, max_tokens=1000, temperature=0.0))
        relevant_resource_types = safe_extract_json(response_str)
        if not isinstance(relevant_resource_types, list):
            print(f"[WARN] Could not decode relevant resource types: {response_str[:200]}")
            relevant_resource_types = []
        relevant_resource_types = [item for item in relevant_resource_types if item in manifest]
        question_lower = user_question.lower()
        fallback_types = []
        if any(term in question_lower for term in ["medication", "medications", "administered", "drug", "drugs"]):
            fallback_types.extend(["MedicationAdministration", "MedicationRequest", "MedicationStatement"])
        if "sepsis" in question_lower:
            fallback_types.extend(["Encounter", "Condition"])
        if _is_sepsis_medication_question(user_question):
            relevant_resource_types = ["Condition", "Encounter", "MedicationAdministration"]
            print(f"[INFO] Using deterministic sepsis medication resources: {', '.join(relevant_resource_types)}")
        elif not relevant_resource_types:
            relevant_resource_types = [item for item in dict.fromkeys(fallback_types) if item in manifest]
            if relevant_resource_types:
                print(f"[INFO] Falling back to resource types from question keywords: {', '.join(relevant_resource_types)}")
        print(f"Relevant resource types: {', '.join(relevant_resource_types)}")
        return {"relevant_resource_types": relevant_resource_types}

    def select_data_to_retrieve(state):
        manifest = state.get("patient_fhir_manifest", {})
        relevant_resource_types = state.get("relevant_resource_types", [])
        tool_calls_to_execute = []
        tools_string = render_text_description([get_patient_fhir_resource])
        for resource_type in relevant_resource_types:
            user_question = state["messages"][1].content
            if resource_type not in manifest and not _is_sepsis_medication_question(user_question):
                print(f"No data found for {resource_type} in the manifest.")
                continue
            manifest_content = f"**{resource_type}**: "
            if len(manifest.get(resource_type, [])) > 0:
                manifest_content += f"Available codes include: {_format_manifest_codes(manifest[resource_type])}\n"
            else:
                manifest_content += "Present (no specific codes found)\n"
            if _is_sepsis_medication_question(user_question) and resource_type in {"Condition", "Encounter", "MedicationAdministration"}:
                patient_id = _patient_id_from_question(user_question)
                tool_call = {
                    "name": "get_patient_fhir_resource",
                    "args": {"patient_id": patient_id, "fhir_resource": resource_type},
                } if patient_id else {}
            else:
                prompt = (
                    "SYSTEM INSTRUCTION: think silently if needed.\n"
                    f"FOR CONTEXT ONLY, USER QUESTION: {state['messages'][1].content}\n\n"
                    f"PATIENT DATA MANIFEST: {manifest_content}\n\n"
                    "You are a specialized API request generator. Your SOLE task is to output a JSON of a tool call to gather the necessary information to answer the user's question. Respond with ONLY a JSON, no explanations or prose.\n"
                    f"Your available tool is:\n{tools_string}\n\n"
                    f"**At this stage you can only call {resource_type}.**\n"
                    "Based on the user question, if the available data in the manifest would be helpful call the tool otherwise output empty JSON {}.\n"
                    "EXAMPLE 1:\n"
                    "{\"name\": \"get_patient_fhir_resource\", \"args\": {\"patient_id\": \"some-patient-id\", \"fhir_resource\": \""
                    f"{resource_type}"
                    "\", \"filter_code\": \"csv-codes-from-manifest\"}},\n"
                    "EXAMPLE 2:\n"
                    "{}"
                )
                response_str = llm_text(llm.invoke(prompt, max_tokens=8000, temperature=0.0))
                tool_call = safe_extract_json(response_str)
            if tool_call and isinstance(tool_call, dict):
                args = tool_call.setdefault("args", {})
                if "patient_id" in args:
                    args["patient_id"] = str(args["patient_id"]).rstrip(".")
                tool_calls_to_execute.append({**tool_call, "id": resource_type})
            else:
                raw_response = locals().get("response_str", "")
                print(f"[WARN] Could not decode tool call for {resource_type}: {str(raw_response)[:200]}")
        print(f"Tool calls to execute: {', '.join(call['id'] for call in tool_calls_to_execute)}")
        return {"tool_calls_to_execute": tool_calls_to_execute}

    def execute_data_retrieval(state):
        concise_facts = []
        raw_tool_outputs = []
        question = state["messages"][1].content
        use_deterministic_sepsis_med_summary = _is_sepsis_medication_question(question)

        for tool_call in state.get("tool_calls_to_execute", []):
            resource_type = tool_call.get("id", "unknown_resource")
            print(f"Fetching and summarizing {resource_type}")
            try:
                tool_output = get_patient_fhir_resource.invoke(tool_call["args"])
            except Exception as exc:
                print(f"[WARN] Tool call failed for {resource_type}: {exc}")
                continue
            raw_tool_outputs.append(tool_output)

            if use_deterministic_sepsis_med_summary:
                continue

            concise_facts_prompt = (
                "SYSTEM INSTRUCTION: think silently if needed.\n"
                f"FOR CONTEXT ONLY, USER QUESTION: {question}\n\n"
                f"TOOL OUTPUT:\n{tool_output}\n\n"
                "You are a fact summarizing agent. Collect facts from TOOL OUTPUT only if they are relevant to answer the USER QUESTION.\n"
                "Write a concise English summary with dates and FHIR references where critical. Do not answer the question.\n"
                "Do not include any thinking or reasoning. Output only the facts in plain text, no markdown or JSON.\n"
            )
            current_summary = llm_text(llm.invoke(concise_facts_prompt, max_tokens=3000, temperature=0.6))
            concise_facts.append(exclude_thinking_component(current_summary))

        if use_deterministic_sepsis_med_summary:
            combined_output = _combine_fhir_bundles(raw_tool_outputs)
            current_summary = _summarize_sepsis_medication_administrations(combined_output)
            if current_summary:
                concise_facts.append(current_summary)

        return {"tool_output_summary": concise_facts}

    def get_final_answer(state):
        summarized_information = "\n\n".join(state["tool_output_summary"])
        if not summarized_information.strip():
            return {"messages": [AIMessage(content="No relevant FHIR resources were retrieved.")]}
        if _is_sepsis_medication_question(state["messages"][1].content):
            return {"messages": [AIMessage(content=summarized_information.strip())]}
        prompt = (
            "Synthesize the following summarized information into a clear, final answer. "
            "Use markdown formatting but DO NOT include any thinking, reasoning, or code blocks.\n\n"
            f"USER QUESTION: {state['messages'][1].content}\n\n"
            f"SUMMARIZED INFORMATION: {summarized_information}\n\n"
            "Final Answer:"
        )
        response = llm_text(llm.invoke(prompt, max_tokens=1500, temperature=0.1))
        response = exclude_thinking_component(response)
        response = response.removesuffix("```").removeprefix("```markdown").strip()
        return {"messages": [AIMessage(content=response)]}

    workflow = StateGraph(AgentState)
    workflow.add_node("call_manifest_tool", call_manifest_tool_node)
    workflow.add_node("identify_relevant_resource_types", identify_relevant_resource_types)
    workflow.add_node("select_data_to_retrieve", select_data_to_retrieve)
    workflow.add_node("execute_data_retrieval", execute_data_retrieval)
    workflow.add_node("final_answer", get_final_answer)
    workflow.set_entry_point("call_manifest_tool")
    workflow.add_edge("call_manifest_tool", "identify_relevant_resource_types")
    workflow.add_edge("identify_relevant_resource_types", "select_data_to_retrieve")
    workflow.add_edge("select_data_to_retrieve", "execute_data_retrieval")
    workflow.add_edge("execute_data_retrieval", "final_answer")
    workflow.add_edge("final_answer", END)
    return workflow.compile()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MedGemma EHR navigator agent.")
    parser.add_argument("--llm_backend", choices=["local", "vertex"], default=os.getenv("MEDGEMMA_LLM_BACKEND", "local"))
    parser.add_argument("--model_path", default=os.getenv("MEDGEMMA_MODEL_PATH"))
    parser.add_argument("--device_map", default=os.getenv("MEDGEMMA_DEVICE_MAP", "auto"))
    parser.add_argument("--torch_dtype", default=os.getenv("MEDGEMMA_TORCH_DTYPE", "auto"))
    parser.add_argument("--project_id", default=os.getenv("GOOGLE_CLOUD_PROJECT", "hai-cd3-foundations"))
    parser.add_argument("--region", default=os.getenv("MEDGEMMA_VERTEX_REGION", "us-central1"))
    parser.add_argument("--endpoint_id", default=os.getenv("MEDGEMMA_VERTEX_ENDPOINT_ID", "1030"))
    parser.add_argument("--fhir_store_url", default=os.getenv("FHIR_STORE_URL", ""))
    parser.add_argument("--fhir_data_dir", default=os.getenv("FHIR_DATA_DIR"))
    parser.add_argument("--patient_id", default=os.getenv("EHR_PATIENT_ID", "auto"))
    parser.add_argument("--question", default="What specific medications were administered to the patient during their sepsis encounter?")
    parser.add_argument("--output_dir", default="./ehr_navigator_outputs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_patient_id_for_local_data(args) -> None:
    if not args.fhir_data_dir or not _is_sepsis_medication_question(args.question):
        return

    requested_patient_id = str(args.patient_id or "auto")
    should_auto_scan = requested_patient_id == "auto" or requested_patient_id == DEMO_SEPSIS_PATIENT_ID
    if not should_auto_scan:
        return

    try:
        from find_sepsis_medication_patient import scan, scan_any_medication_patient
    except Exception as exc:
        print(f"[WARN] Could not import FHIR patient scanner: {exc}")
        return

    fhir_data_dir = Path(args.fhir_data_dir)
    print(f"Scanning FHIR data for a patient with sepsis-related medication administrations in {args.fhir_data_dir}")
    try:
        first_match = next(scan(fhir_data_dir))
        _, patient_id, matches = first_match
        if patient_id != requested_patient_id:
            print(f"[INFO] Using local FHIR patient {patient_id} with {len(matches)} sepsis-related medication administration(s).")
        args.patient_id = patient_id
        return
    except StopIteration:
        pass

    try:
        first_match = next(scan_any_medication_patient(fhir_data_dir))
    except StopIteration:
        message = (
            "No patient with MedicationAdministration records was found "
            f"in the configured FHIR data directory: {args.fhir_data_dir}"
        )
        if requested_patient_id == "auto":
            raise SystemExit(message)
        print(f"[WARN] {message}. Continuing with requested patient {requested_patient_id}.")
        return

    _, patient_id, matches = first_match
    print(
        "[WARN] No sepsis-related MedicationAdministration patient was found. "
        f"Falling back to local patient {patient_id} with {len(matches)} MedicationAdministration record(s)."
    )
    args.patient_id = patient_id
    args.question = "What medications were administered to the patient?"

def main():
    args = parse_args()
    _resolve_patient_id_for_local_data(args)
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.llm_backend == "vertex":
        from langchain_google_vertexai import VertexAIModelGarden
        print("Initializing Vertex AI Model Garden MedGemma endpoint")
        llm = VertexAIModelGarden(
            project=args.project_id,
            location=args.region,
            endpoint_id=args.endpoint_id,
            allowed_model_args=["temperature", "max_tokens"],
        )
    else:
        model_path = args.model_path
        if not model_path:
            datasets_dir = os.getenv("ONESCIENCE_DATASETS_DIR", "")
            model_path = str(Path(datasets_dir) / "medgemma/model_garden/google--medgemma-27b-text-it/snapshots/master")
        print(f"Initializing local MedGemma model from {model_path}")
        llm = LocalMedGemmaLLM(model_path, device_map=args.device_map, torch_dtype=args.torch_dtype)

    agent = build_agent(llm, args.fhir_store_url, local_fhir_data_dir=args.fhir_data_dir, verbose=args.verbose)

    composed_question = f"{args.question}. Patient ID {args.patient_id}."
    messages = [SystemMessage(content="You are MedGemma, a helpful, expert medical assistant."), HumanMessage(content=composed_question)]
    inputs = {
        "messages": messages,
        "patient_fhir_manifest": {},
        "tool_output_summary": [],
        "tool_calls_to_execute": [],
        "relevant_resource_types": [],
    }

    print(f"Invoking EHR navigator agent for patient {args.patient_id}")
    final_state = agent.invoke(inputs)
    final_response = final_state["messages"][-1].content
    final_response = str(final_response).removesuffix("```").removeprefix("```markdown").strip()

    result = {
        "project_id": args.project_id,
        "region": args.region,
        "endpoint_id": args.endpoint_id,
        "fhir_store_url": args.fhir_store_url,
        "patient_id": args.patient_id,
        "question": args.question,
        "relevant_resource_types": final_state.get("relevant_resource_types", []),
        "tool_output_summary": final_state.get("tool_output_summary", []),
        "final_answer": final_response,
    }

    json_path = output_dir / "ehr_navigator_result.json"
    md_path = output_dir / "ehr_navigator_result.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(f"# Agent Final Answer\n\n{final_response}\n", encoding="utf-8")

    print(f"Saved JSON result to {json_path}")
    print(f"Saved markdown result to {md_path}")
    print("\nFinal answer:\n")
    print(final_response)


if __name__ == "__main__":
    main()