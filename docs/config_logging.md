# Config Logging System

## Overview

Das FIT-Package protokolliert automatisch alle Konfigurationsparameter bei jedem Experiment. Dies gewährleistet vollständige Reproduzierbarkeit und Nachvollziehbarkeit aller Experimente.

## Automatisches Logging

Beim Import von `FIT_python.config` wird automatisch ein Config-Snapshot erstellt:

```python
import FIT_python.config as cfg
# --> Erstellt automatisch: <experiment_root>/config_logs/config_<timestamp>_<hash>_initial.json
```

### Was wird geloggt?

- **Timestamp**: Wann wurde die Config erstellt/geändert
- **Experiment Name**: Wert von `FIT_EXPERIMENT_NAME` Environment Variable
- **Pfade**: Alle wichtigen Verzeichnisse (RAW_DIR, EXPERIMENT_ROOT, etc.)
- **Parameter**: Alle Werte aus dem `CONFIG` Dictionary
- **Seeds & Settings**: GLOBAL_RANDOM_SEED, NUM_FOLDS, DEBUG_MODE, etc.

### Dateistruktur

```
<experiment_root>/
  config_logs/
    config_20260120_171747_14f95bdd_initial.json      # Initiale Config
    config_20260120_172530_a3b2c1d0_manual_update.json # Nach manueller Änderung
    config_20260120_173045_e5f6g7h8_modified.json     # Automatisch erkannte Änderung
    config_latest.json                                 # Symlink zur aktuellsten Config
```

## Manuelle Snapshots

Wenn du Parameter im Notebook änderst, solltest du einen Snapshot erstellen:

```python
import FIT_python.config as cfg

# Parameter ändern
cfg.CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 12
cfg.NUM_FOLDS = 10
cfg.GLOBAL_RANDOM_SEED = 456

# Snapshot mit Beschreibung erstellen
cfg.log_config_change("increased_trail_size_and_folds")
# --> Erstellt: config_logs/config_20260120_172530_a3b2c1d0_increased_trail_size_and_folds.json
```

## Change Detection

Das System erkennt automatisch Änderungen durch Hash-Vergleich:

- Bei jedem Import wird die aktuelle Config mit `config_latest.json` verglichen
- Wenn sich etwas geändert hat, wird automatisch ein neuer Snapshot mit Grund "modified" erstellt
- Der Hash basiert auf allen Config-Werten (außer Timestamp)

## Best Practices

### 1. Im Notebook: Änderungen sofort dokumentieren

```python
# SCHLECHT:
cfg.CONFIG["something"] = new_value
# ... viele Zellen später ...
# Welche Parameter hatte ich nochmal geändert?

# GUT:
cfg.CONFIG["pipeline_individual_id"]["trail_generation_defaults"]["sample_size"] = 12
cfg.log_config_change("testing_larger_trail_size")
```

### 2. Aussagekräftige Gründe angeben

```python
# SCHLECHT:
cfg.log_config_change("test")
cfg.log_config_change("update")

# GUT:
cfg.log_config_change("doubled_trail_size_for_better_coverage")
cfg.log_config_change("reduced_folds_to_speed_up_debugging")
cfg.log_config_change("changed_seed_for_different_splits")
```

### 3. Config-Logs mit committen (bei wichtigen Experimenten)

Die `config_logs/` sollten mit ins Repository, besonders bei:
- Finalen Experimenten für die Dissertation
- Experimenten mit guten Ergebnissen
- Baseline-Runs zum Vergleichen

## Config-Snapshots vergleichen

```python
import json
from pathlib import Path

config_dir = Path("results/my_experiment/config_logs")

# Zwei Snapshots laden
with open(config_dir / "config_20260120_171747_14f95bdd_initial.json") as f:
    config1 = json.load(f)
    
with open(config_dir / "config_20260120_172530_a3b2c1d0_manual_update.json") as f:
    config2 = json.load(f)

# Änderungen finden (vereinfacht)
def find_differences(d1, d2, path=""):
    for key in set(d1.keys()) | set(d2.keys()):
        new_path = f"{path}.{key}" if path else key
        if key not in d1:
            print(f"+ {new_path}: {d2[key]}")
        elif key not in d2:
            print(f"- {new_path}: {d1[key]}")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            find_differences(d1[key], d2[key], new_path)
        elif d1[key] != d2[key]:
            print(f"~ {new_path}: {d1[key]} -> {d2[key]}")

find_differences(config1["config"], config2["config"])
```

## Troubleshooting

### "Config wird nicht geloggt"

- Stelle sicher, dass `EXPERIMENT_ROOT` existiert und beschreibbar ist
- Prüfe die Konsolen-Ausgabe beim Import: `[CONFIG] Initial config saved to: ...`

### "Zu viele Config-Files"

Das ist normal und gewollt! Jede Änderung wird protokolliert. Falls nötig:
- Alte Snapshots können gelöscht werden (außer wichtige Experiment-Configs)
- `config_latest.json` zeigt immer den aktuellen Stand

### "Hash ändert sich ständig"

Der Timestamp wird bei der Hash-Berechnung ignoriert, also sollte der Hash nur bei tatsächlichen Config-Änderungen unterschiedlich sein.

## Beispiel: Config-Log nach Experiment

```json
{
  "timestamp": "2026-01-20T17:17:47.123456",
  "experiment_name": "fit_notebook_final_2026",
  "experiment_root": "c:\\otter_fit\\FIT_package\\results\\fit_notebook_final_2026",
  "raw_dir": "c:\\otter_fit\\FIT_package\\data\\raw",
  "global_random_seed": 123,
  "num_folds": 5,
  "debug_mode": false,
  "config": {
    "pipeline_individual_id": {
      "trail_generation_defaults": {
        "sample_size": 9,
        "subsample_sizes": [3, 5, 7],
        "n_candidates": 20
      },
      ...
    },
    ...
  },
  "paths": {
    "experiment_root": "c:\\otter_fit\\FIT_package\\results\\fit_notebook_final_2026",
    "splits": "c:\\otter_fit\\FIT_package\\results\\fit_notebook_final_2026\\data\\splits",
    ...
  }
}
```
