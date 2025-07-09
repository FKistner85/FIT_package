from __future__ import annotations
from pathlib import Path

def save_caption(fig_path: Path | str, caption: str) -> None:
    """Save `caption` alongside a figure.

    The caption is written to a text file with the same stem as ``fig_path``.
    ``fig_path`` may be a string or :class:`~pathlib.Path`.
    """
    path = Path(fig_path)
    txt_path = path.with_suffix(path.suffix + '.txt')
    txt_path.write_text(caption, encoding='utf-8')

