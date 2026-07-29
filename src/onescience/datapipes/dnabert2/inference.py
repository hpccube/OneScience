"""Input readers for DNABERT-2 embedding inference."""

import csv
from dataclasses import dataclass
from pathlib import Path

from onescience.utils.dnabert2 import normalize_dna


@dataclass(frozen=True)
class SequenceRecord:
    """A named DNA sequence."""

    identifier: str
    sequence: str


def _read_fasta(path: Path) -> list[SequenceRecord]:
    records = []
    identifier = None
    sequence_parts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append(SequenceRecord(identifier, "".join(sequence_parts)))
            identifier = line[1:].strip() or f"sequence_{len(records)}"
            sequence_parts = []
        elif identifier is None:
            raise ValueError(f"FASTA sequence found before header in {path}")
        else:
            sequence_parts.append(line)
    if identifier is not None:
        records.append(SequenceRecord(identifier, "".join(sequence_parts)))
    return records


def _read_csv(path: Path) -> list[SequenceRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sequence" not in reader.fieldnames:
            raise ValueError(f"CSV input must contain a 'sequence' column: {path}")
        return [
            SequenceRecord(row.get("id") or f"sequence_{index}", row["sequence"])
            for index, row in enumerate(reader)
        ]


def read_sequence_records(path: str | Path) -> list[SequenceRecord]:
    """Read FASTA, CSV, or one-sequence-per-line text input."""
    input_path = Path(path)
    if input_path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        records = _read_fasta(input_path)
    elif input_path.suffix.lower() == ".csv":
        records = _read_csv(input_path)
    else:
        records = [
            SequenceRecord(f"sequence_{index}", line.strip())
            for index, line in enumerate(input_path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        ]
    if not records:
        raise ValueError(f"No DNA sequences found in {input_path}")
    return [
        SequenceRecord(record.identifier, normalize_dna(record.sequence))
        for record in records
    ]
