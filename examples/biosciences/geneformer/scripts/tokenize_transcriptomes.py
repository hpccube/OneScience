"""Tokenize raw-count Loom, AnnData, or Zarr transcriptomes for Geneformer."""

import argparse
from pathlib import Path

from onescience.datapipes.geneformer import TranscriptomeTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="tokenized_cells")
    parser.add_argument("--file-format", choices=("loom", "h5ad", "zarr"), default="h5ad")
    parser.add_argument("--model-version", choices=("V1", "V2"), default="V2")
    parser.add_argument("--metadata", action="append", default=[], metavar="INPUT=OUTPUT")
    parser.add_argument("--input-identifier", default="")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--use-h5ad-index", action="store_true")
    parser.add_argument("--keep-counts", action="store_true")
    parser.add_argument("--no-collapse-gene-ids", action="store_true")
    parser.add_argument("--use-generator", action="store_true")
    return parser.parse_args()


def parse_metadata(values: list[str]) -> dict[str, str] | None:
    if not values:
        return None
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid metadata mapping {value!r}; expected INPUT=OUTPUT")
        source, destination = value.split("=", 1)
        result[source] = destination
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict=parse_metadata(args.metadata),
        nproc=args.nproc,
        chunk_size=args.chunk_size,
        collapse_gene_ids=not args.no_collapse_gene_ids,
        use_h5ad_index=args.use_h5ad_index,
        keep_counts=args.keep_counts,
        model_version=args.model_version,
    )
    tokenizer.tokenize_data(
        data_directory=args.input_dir,
        output_directory=args.output_dir,
        output_prefix=args.output_prefix,
        file_format=args.file_format,
        input_identifier=args.input_identifier,
        use_generator=args.use_generator,
    )
    print(f"Saved tokenized data to {args.output_dir / (args.output_prefix + '.dataset')}")


if __name__ == "__main__":
    main()
