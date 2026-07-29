"""DNA sequence normalization helpers used by DNABERT-2 data pipes."""

IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")
_COMPLEMENT = str.maketrans(
    "ACGTRYSWKMBDHVN",
    "TGCAYRSWMKVHDBN",
)


def normalize_dna(sequence: str, allow_ambiguous: bool = True) -> str:
    """Normalize whitespace/case and validate a DNA sequence."""
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("DNA sequence must not be empty")
    alphabet = IUPAC_DNA if allow_ambiguous else frozenset("ACGT")
    invalid = sorted(set(normalized) - alphabet)
    if invalid:
        raise ValueError(f"Unsupported DNA symbols: {''.join(invalid)}")
    return normalized


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of an IUPAC DNA sequence."""
    normalized = normalize_dna(sequence)
    return normalized.translate(_COMPLEMENT)[::-1]


def generate_kmer_text(sequence: str, k: int) -> str:
    """Convert a DNA sequence into the overlapping k-mer text format."""
    normalized = normalize_dna(sequence)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > len(normalized):
        raise ValueError(f"k={k} exceeds sequence length {len(normalized)}")
    return " ".join(normalized[index : index + k] for index in range(len(normalized) - k + 1))
