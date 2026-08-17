"""Package data for the scGPT model implementation."""


def get_package_data():
    return {"onescience.models.scgpt": ["LICENSE.scgpt"]}


def get_manifest_rules():
    return ["include src/onescience/models/scgpt/LICENSE.scgpt"]
