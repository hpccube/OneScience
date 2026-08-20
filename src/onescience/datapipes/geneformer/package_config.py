"""Package-data declaration for Geneformer dictionaries."""


def get_package_data():
    return {
        "onescience.datapipes.geneformer": [
            "data/*.pkl",
            "data/gene_dictionaries_30m/*.pkl",
        ],
    }


def get_manifest_rules():
    return [
        "recursive-include src/onescience/datapipes/geneformer/data *.pkl",
    ]
