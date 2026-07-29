"""Supervised sequence classification data pipe for DNABERT-2."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from onescience.utils.dnabert2 import generate_kmer_text, normalize_dna


@dataclass(frozen=True)
class ClassificationExample:
    sequence: str
    label: int
    sequence_pair: str | None = None


class SequenceClassificationDataset(Dataset):
    """Load official two- or three-column DNABERT classification CSV files."""

    def __init__(self, path: str | Path, num_labels: int | None = None):
        self.path = Path(path)
        with self.path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            raise ValueError(f"Dataset must contain a header and samples: {self.path}")

        self.examples = []
        width = len(rows[0])
        if width not in {2, 3}:
            raise ValueError("Expected sequence,label or sequence1,sequence2,label CSV")
        for row_index, row in enumerate(rows[1:], start=2):
            if len(row) != width:
                raise ValueError(f"Inconsistent column count at {self.path}:{row_index}")
            try:
                label = int(row[-1].strip())
            except ValueError as error:
                raise ValueError(f"Label must be an integer at {self.path}:{row_index}") from error
            sequence = normalize_dna(row[0])
            sequence_pair = normalize_dna(row[1]) if width == 3 else None
            self.examples.append(ClassificationExample(sequence, label, sequence_pair))

        labels = sorted({example.label for example in self.examples})
        if num_labels is None:
            if labels != list(range(len(labels))):
                raise ValueError(f"Labels must be contiguous integers starting at 0, found {labels}")
            self.num_labels = len(labels)
        else:
            if any(label < 0 or label >= num_labels for label in labels):
                raise ValueError(f"Labels {labels} fall outside configured range [0, {num_labels})")
            self.num_labels = num_labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ClassificationExample:
        return self.examples[index]


@dataclass
class SequenceBatchCollator:
    """Tokenize single or paired DNA sequences at batch time."""

    tokenizer: Any
    max_length: int = 512
    kmer: int = -1

    def __call__(self, instances: Sequence[ClassificationExample]) -> dict[str, torch.Tensor]:
        sequences = [self._prepare(instance.sequence) for instance in instances]
        pairs = [
            self._prepare(instance.sequence_pair) if instance.sequence_pair is not None else None
            for instance in instances
        ]
        has_pairs = any(pair is not None for pair in pairs)
        if has_pairs and not all(pair is not None for pair in pairs):
            raise ValueError("A batch cannot mix single-sequence and sequence-pair samples")
        encoded = self.tokenizer(
            sequences,
            text_pair=pairs if has_pairs else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([instance.label for instance in instances], dtype=torch.long)
        return encoded

    def _prepare(self, sequence: str) -> str:
        if self.kmer == -1:
            return sequence
        return generate_kmer_text(sequence, self.kmer)
