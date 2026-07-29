"""Package data configuration for the bundled DNABERT-2 architecture."""

DNABERT2_PACKAGE_DATA = {
    "onescience.models.dnabert2.hf_architecture": [
        "config.json",
    ],
}

DNABERT2_MANIFEST_RULES = [
    "include src/onescience/models/dnabert2/hf_architecture/config.json",
]


def get_package_data():
    return DNABERT2_PACKAGE_DATA


def get_manifest_rules():
    return DNABERT2_MANIFEST_RULES
