from __future__ import annotations

from pathlib import Path
from typing import Iterable

import FIT_python.config as config


def _species_dirs(species: str, base: Path) -> Path | None:
    """Return the directory for ``species`` under ``base`` if it exists."""
    candidates = [
        species.replace("_", " ").title().replace(" ", "_"),
        species.capitalize(),
        species,
    ]
    for cand in candidates:
        path = base / cand
        if path.is_dir():
            return path
    return None

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


def _build_categories() -> list[tuple[str, list[Path]]]:
    """Assemble figure categories dynamically for all available species."""
    categories: list[tuple[str, list[Path]]] = []

    # Generic experiment level figures
    split_dir = (
        config.RESULTS_DIR / 'experiments' / 'fit_start_to_finish' / 'split_fig'
    )
    if split_dir.exists():
        categories.append(('Data loading and split', [split_dir]))

    sex_boxplot = (
        config.RESULTS_DIR / 'experiments' / 'fit_start_to_finish' / 'sex_boxplots.png'
    )
    if sex_boxplot.exists():
        categories.append(('Sex model Evaluation', [sex_boxplot]))

    # Species specific figures
    splits_root = config.DATA_DIR / 'splits'
    species_list = sorted(p.name for p in splits_root.iterdir() if p.is_dir())
    for species in species_list:
        paths: list[Path] = []

        dist_dir = _species_dirs(
            species, config.RESULTS_DIR / 'figures' / 'feature_distributions'
        )
        if dist_dir:
            paths.append(dist_dir)

        corr_dir = _species_dirs(
            species, config.RESULTS_DIR / 'figures' / 'feature_correlations'
        )
        if corr_dir:
            paths.append(corr_dir)

        id_dir = (
            config.RESULTS_DIR
            / 'experiments'
            / 'fit_start_to_finish'
            / 'id_baseline'
            / species
        )
        if id_dir.is_dir():
            paths.append(id_dir)

        if paths:
            title = species.replace('_', ' ').title()
            categories.append((title, paths))

    return categories

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
    for title, paths in _build_categories():
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
