#!/usr/bin/env python3
"""Synchronize requirements.txt and environment.yml from pyproject.toml."""
from __future__ import annotations
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
ENV_FILE = ROOT / "environment.yml"


def parse_pyproject() -> tuple[list[str], list[str], list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    deps: list[str] = data["project"].get("dependencies", [])
    extras: dict[str, list[str]] = data["project"].get("optional-dependencies", {})
    dev = extras.get("dev", [])
    torch = extras.get("torch", [])
    return deps, dev, torch


def write_requirements(deps: list[str], dev: list[str], torch: list[str]) -> None:
    lines = [
        "# Auto-generated from pyproject.toml", 
        "# Core runtime packages",
        *deps,
        "",
        "# Development and testing tools",
        *dev,
    ]
    if torch:
        lines += ["", "# Optional torch support", *torch]
    lines += ["", "# Install local package in editable mode", "-e ."]
    REQUIREMENTS.write_text("\n".join(lines) + "\n")


def write_environment(deps: list[str], dev: list[str], torch: list[str]) -> None:
    lines = [
        "name: FIT_python_conda_env",
        "",
        "channels:",
        "  - conda-forge",
        "  - defaults",
        "",
        "dependencies:",
        "  - python=3.11",
    ]
    for dep in deps:
        lines.append(f"  - {dep}")
    lines += [
        "",
        "  - pip",
        "  - pip:",
    ]
    for item in ["-e .", *dev]:
        lines.append(f"    - {item}")
    ENV_FILE.write_text("\n".join(lines) + "\n")


def main() -> None:
    deps, dev, torch = parse_pyproject()
    write_requirements(deps, dev, torch)
    write_environment(deps, dev, torch)


if __name__ == "__main__":
    main()
