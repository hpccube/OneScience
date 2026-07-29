from pathlib import Path


def get_package_data():
    package = "onescience.models.mattergen"
    data = {
        package: ["LICENSE", "NOTICE", "SOURCE.md", "**/*.yaml", "**/*.json"],
    }
    return data
