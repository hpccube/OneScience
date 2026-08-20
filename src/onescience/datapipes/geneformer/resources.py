"""Paths to the Geneformer vocabulary and gene-normalization resources."""

from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"

GENE_MEDIAN_FILE = DATA_DIR / "gene_median_dictionary_gc104M.pkl"
TOKEN_DICTIONARY_FILE = DATA_DIR / "token_dictionary_gc104M.pkl"
ENSEMBL_DICTIONARY_FILE = DATA_DIR / "gene_name_id_dict_gc104M.pkl"
ENSEMBL_MAPPING_FILE = DATA_DIR / "ensembl_mapping_dict_gc104M.pkl"

GENE_DICTIONARIES_30M_DIR = DATA_DIR / "gene_dictionaries_30m"
GENE_MEDIAN_FILE_30M = (
    GENE_DICTIONARIES_30M_DIR / "gene_median_dictionary_gc30M.pkl"
)
TOKEN_DICTIONARY_FILE_30M = GENE_DICTIONARIES_30M_DIR / "token_dictionary_gc30M.pkl"
ENSEMBL_DICTIONARY_FILE_30M = (
    GENE_DICTIONARIES_30M_DIR / "gene_name_id_dict_gc30M.pkl"
)
ENSEMBL_MAPPING_FILE_30M = (
    GENE_DICTIONARIES_30M_DIR / "ensembl_mapping_dict_gc30M.pkl"
)

__all__ = [
    "DATA_DIR",
    "ENSEMBL_DICTIONARY_FILE",
    "ENSEMBL_DICTIONARY_FILE_30M",
    "ENSEMBL_MAPPING_FILE",
    "ENSEMBL_MAPPING_FILE_30M",
    "GENE_MEDIAN_FILE",
    "GENE_MEDIAN_FILE_30M",
    "TOKEN_DICTIONARY_FILE",
    "TOKEN_DICTIONARY_FILE_30M",
]
