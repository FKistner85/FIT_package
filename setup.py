"""Setup configuration for the FIT_python package."""

from __future__ import annotations

import tomllib
from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).resolve().parent
with open(root / "pyproject.toml", "rb") as f:
    config = tomllib.load(f)

project = config["project"]
install_requires = project.get("dependencies", [])
extras_require = project.get("optional-dependencies", {})

setup(
    name=project["name"],
    version=project["version"],
    description=project["description"],
    author=project["authors"][0]["name"],

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=project["requires-python"],
    install_requires=install_requires,
    extras_require=extras_require,
)
