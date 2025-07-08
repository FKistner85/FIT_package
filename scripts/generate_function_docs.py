from __future__ import annotations

"""Generate markdown documentation for internal functions/classes and external imports."""

import ast
from pathlib import Path
from typing import List, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "FIT_python"
PACKAGE_NAME = "FIT_python"


def gather_docs() -> Tuple[List[Tuple[str, str]], List[str]]:
    internal_docs: List[Tuple[str, str]] = []
    external_imports: set[str] = set()

    for path in PACKAGE_ROOT.rglob("*.py"):
        rel_module = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
        module_name = ".".join(rel_module.parts)

        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node) or ""
                internal_docs.append((f"{module_name}.{node.name}", doc))

        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith(PACKAGE_NAME):
                        external_imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith(PACKAGE_NAME):
                    external_imports.add(node.module.split(".")[0])

    internal_docs.sort(key=lambda t: t[0])
    return internal_docs, sorted(external_imports)


def write_docs(int_docs: List[Tuple[str, str]], ext_mods: List[str]) -> None:
    docs_en = Path("docs/EN")
    docs_de = Path("docs/DE")
    docs_en.mkdir(parents=True, exist_ok=True)
    docs_de.mkdir(parents=True, exist_ok=True)

    def _write_lang(path: Path, entries: List[Tuple[str, str]]):
        lines = ["# Internal Functions and Classes", ""]
        for qualname, doc in entries:
            lines.append(f"## {qualname}")
            if doc:
                lines.append(doc)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    _write_lang(docs_en / "own_functions.md", int_docs)
    _write_lang(docs_de / "own_functions.md", int_docs)

    lines = ["# External Imports", ""]
    for mod in ext_mods:
        lines.append(f"- {mod}")
    (docs_en / "external_imports.md").write_text("\n".join(lines), encoding="utf-8")
    (docs_de / "external_imports.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    int_docs, ext_mods = gather_docs()
    write_docs(int_docs, ext_mods)


if __name__ == "__main__":
    main()
