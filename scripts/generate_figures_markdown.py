from __future__ import annotations

from pathlib import Path
from typing import Iterable

import FIT_python.config as config

PLOT_TYPES: dict[str, str] = {
    'boxplot': 'Boxplot',
    'heatmap': 'Heatmap',
    'heatmaps': 'Heatmap',
    'scatter': 'Scatter plot',
    'summary': 'Summary',
    'probabilities': 'Probability plot',
    'prediction': 'Prediction',
    'hyperparam': 'Hyperparameter plot',
    'stepimpact': 'Step impact',
}

OUTPUT_FILE = config.EXPERIMENT_ROOT / 'docs' / 'generated_figures.md'

# Mapping of figure categories to the directories or files containing images
CATEGORIES: list[tuple[str, list[Path]]] = [
    (
        'Data loading and split',
        [
            config.RESULTS_DIR / 'figures' / 'summary',
            config.RESULTS_DIR / 'data' / 'eurasian_otter_fig',
        ],
    ),
    (
        'Sex model Evaluation',
        [
            config.RESULTS_DIR / 'figures' / 'sex_model',
            config.RESULTS_DIR / 'data' / 'eurasian_otter_random_search_standard_metrics',
            config.RESULTS_DIR / 'figures' / 'raw_balanced_acc_heatmap_min_max_highlight.png',
        ],
    ),
    (
        'Individual ID',
        [
            config.RESULTS_DIR / 'figures' / 'plots_otter',
            config.RESULTS_DIR / 'figures' / 'plots otter',
        ],
    ),
]

def _iter_images(paths: Iterable[Path]):
    for p in paths:
        if p.is_dir():
            for ext in ('*.png', '*.svg'):
                for img in sorted(p.glob(ext)):
                    yield img
        else:
            if p.exists() and p.suffix.lower() in ('.png', '.svg'):
                yield p

def _caption_for(img: Path) -> str:
    txt = img.with_suffix('.txt')
    if txt.exists():
        return txt.read_text().strip()
    stem = img.stem.lower().replace('__', '_')
    tokens = stem.split('_')
    for token in tokens:
        if token in PLOT_TYPES:
            return f"**{PLOT_TYPES[token]}**"
    return f"**{tokens[-1].replace('-', ' ').title()}**"


def main() -> None:
    lines = ['# Generated Figures', '']
    for title, paths in CATEGORIES:
        images = list(_iter_images(paths))
        if not images:
            continue
        lines.append(f'## {title}')
        lines.append('')
        for img in images:
            caption = _caption_for(img)
            rel = img.as_posix()
            lines.append(f'![{caption}]({rel})')
            lines.append('')
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
