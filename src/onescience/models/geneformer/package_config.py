"""Package-data declaration for the Geneformer third-party license."""


def get_package_data():
    return {"onescience.models.geneformer": ["LICENSE.geneformer"]}


def get_manifest_rules():
    return ["include src/onescience/models/geneformer/LICENSE.geneformer"]
