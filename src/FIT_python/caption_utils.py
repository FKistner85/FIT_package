from __future__ import annotations

from pathlib import Path


def save_caption(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` with ``.txt`` suffix and print the caption."""
    txt_path = Path(path).with_suffix(".txt")
    txt_path.write_text(text)
    print(f"[CAPTION] {text}")

