"""Simple PyQt based GUI for landmark annotation."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
import math
import numpy as np

from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.gui_annotator import image_manager
from FIT_python.gui_annotator.annotation_canvas import AnnotationCanvas, Landmark

CFG = SOFT_CONFIG.get("gui_annotator", {})
RAW_DIR = Path(CFG.get("raw_image_dir", "data/raw/images"))
ANNOTATION_DIR = Path(CFG.get("annotation_dir", "data/processed/annotations"))

# Reference template with measurement definitions
REF_FILE = (
    Path(CFG.get("reference_template_dir", "data/raw/reference_templates"))
    / "landmarks_extended.json"
)
try:
    with REF_FILE.open("r", encoding="utf-8") as fh:
        _REF_DATA = json.load(fh)
except Exception:
    _REF_DATA = {}

VARIABLES = _REF_DATA.get("variables", {})

# Pairs of manual landmark indices (0-based) used for derived points
_DERIVED_PAIRS = [
    (0, 1),  # -> landmark 12
    (1, 2),  # -> landmark 13
    (2, 3),  # -> landmark 14
    (3, 4),  # -> landmark 15
    (5, 6),  # -> landmark 16
    (6, 7),  # -> landmark 17
]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def _angle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Return angle ABC in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = math.hypot(*ba)
    norm_bc = math.hypot(*bc)
    if norm_ba == 0 or norm_bc == 0:
        return float("nan")
    cos = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _triangle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    cx, cy = c
    return 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))


def _midpoint(p1: tuple[float, float], p2: tuple[float, float]) -> list[float]:
    x1, y1 = p1
    x2, y2 = p2
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _compute_derived(landmarks: list[list[float]]) -> list[list[float]]:
    if len(landmarks) != 11:
        raise ValueError(f"expected 11 manual landmarks, got {len(landmarks)}")
    return [_midpoint(landmarks[i], landmarks[j]) for i, j in _DERIVED_PAIRS]


def _compute_measurements(lms: list[list[float]]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for name, spec in VARIABLES.items():
        idxs = [i - 1 for i in spec.get("landmarks", [])]
        if any(i >= len(lms) for i in idxs):
            feats[name] = float("nan")
            continue
        pts = [tuple(lms[i]) for i in idxs]
        if spec.get("type") == "distance":
            feats[name] = _distance(pts[0], pts[1])
        elif spec.get("type") == "angle":
            feats[name] = _angle(pts[0], pts[1], pts[2])
        elif spec.get("type") == "triangle":
            feats[name] = _triangle(pts[0], pts[1], pts[2])
        else:
            feats[name] = float("nan")
    return feats



class AnnotatorApp(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FIT Image Annotator")
        self.canvas = AnnotationCanvas()
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.open_button = QtWidgets.QPushButton("Open Image")
        self.save_button = QtWidgets.QPushButton("Save Annotations")
        self.next_button = QtWidgets.QPushButton("Next")
        self.back_button = QtWidgets.QPushButton("Back")
        for btn in (
            self.open_button,
            self.save_button,
            self.next_button,
            self.back_button,
        ):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
            )
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.canvas.crossfade_opacity * 100))
        self.reference_combo = QtWidgets.QComboBox()
        if self.canvas.reference_names:
            self.reference_combo.addItems(self.canvas.reference_names)
        self.rotate_button = QtWidgets.QPushButton("Rotate")
        self.rotate_button.setCheckable(True)
        self.scale_button = QtWidgets.QPushButton("Scale")
        self.scale_button.setCheckable(True)
        self.landmark_button = QtWidgets.QPushButton("Landmark")
        self.landmark_button.setCheckable(True)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for btn in (self.rotate_button, self.scale_button, self.landmark_button):
            self.mode_group.addButton(btn)
        self.rotate_button.setChecked(True)
        self.scale_label = QtWidgets.QLabel("Scale: n/a")
        self.stats_label = QtWidgets.QLabel("")
        self.landmark_table = QtWidgets.QTableWidget(11, 3)
        self.landmark_table.setHorizontalHeaderLabels(["X", "Y", "Visible"])
        self.landmark_table.setVerticalHeaderLabels([f"LM{i+1}" for i in range(11)])
        self.landmark_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.landmark_table.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.back_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.save_button)
        controls.addStretch()
        layout.addLayout(controls)
        overlay_controls = QtWidgets.QHBoxLayout()
        overlay_controls.addWidget(QtWidgets.QLabel("Overlay:"))
        overlay_controls.addWidget(self.reference_combo)
        overlay_controls.addWidget(QtWidgets.QLabel("Opacity:"))
        overlay_controls.addWidget(self.opacity_slider)
        layout.addLayout(overlay_controls)
        mode_controls = QtWidgets.QHBoxLayout()
        mode_controls.addWidget(self.rotate_button)
        mode_controls.addWidget(self.scale_button)
        mode_controls.addWidget(self.landmark_button)
        mode_controls.addWidget(self.scale_label)
        mode_controls.addStretch()
        layout.addLayout(mode_controls)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.landmark_table)
        self.current_id: str | None = None
        self.image_paths: list[Path] = []
        self.current_index: int | None = None
        self.next_button.setEnabled(False)
        self.back_button.setEnabled(False)
        self.open_button.clicked.connect(self.on_open)
        self.save_button.clicked.connect(self.on_save)
        self.next_button.clicked.connect(self.on_next)
        self.back_button.clicked.connect(self.on_back)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.reference_combo.currentIndexChanged.connect(self._on_reference_changed)
        self.rotate_button.toggled.connect(
            lambda checked: checked and self.canvas.set_mode("rotate")
        )
        self.scale_button.toggled.connect(
            lambda checked: checked and self.canvas.set_mode("scale")
        )
        self.landmark_button.toggled.connect(
            lambda checked: checked and self.canvas.set_mode("landmark")
        )
        self.canvas.scale_changed.connect(self._on_scale_changed)
        self.canvas.landmarks_changed.connect(self._update_landmark_table)

        self._update_landmark_table()

    # slots --------------------------------------------------------------
    def on_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", str(RAW_DIR))
        if not path:
            return
        if not self.image_paths:
            self.image_paths = sorted(RAW_DIR.glob("*.jpg"))
        try:
            self.current_index = self.image_paths.index(Path(path))
        except ValueError:
            self.image_paths.insert(0, Path(path))
            self.current_index = 0
        self._load_current()
        self._update_buttons()

    def on_save(self) -> None:
        if not self.current_id:
            return
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ANNOTATION_DIR / f"{self.current_id}_{ts}.json"
        manual_lms = self.canvas.get_unrotated_landmarks()
        landmarks = [
            {"x": lm.x, "y": lm.y, "visible": lm.visible} for lm in manual_lms
        ]
        coords = [[lm.x, lm.y] for lm in manual_lms]
        if len(coords) == 11:
            derived = _compute_derived(coords)
            for x, y in derived:
                landmarks.append({"x": x, "y": y, "visible": True})
            coords += derived
        data = {
            "image_path": str(RAW_DIR / f"{self.current_id}.jpg"),
            "rotation_deg": math.degrees(self.canvas.rotation_angle),
            "landmarks": landmarks,
            "measurements": _compute_measurements(coords),
        }
        meta_path = RAW_DIR / f"{self.current_id}.jpg"
        data["metadata"] = image_manager.extract_metadata(meta_path)
        if self.canvas.pixels_per_cm is not None:
            data["pixels_per_cm"] = self.canvas.pixels_per_cm
            if self.canvas.pixels_per_cm_sd is not None:
                data["pixels_per_cm_sd"] = self.canvas.pixels_per_cm_sd
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        QtWidgets.QMessageBox.information(self, "Saved", f"Annotations written to {out_path}")

    def on_next(self) -> None:
        if self.current_index is None:
            return
        if self.current_index + 1 < len(self.image_paths):
            self.current_index += 1
            self._load_current()
        self._update_buttons()

    def on_back(self) -> None:
        if self.current_index is None:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current()
        self._update_buttons()

    def _on_opacity_changed(self, value: int) -> None:
        self.canvas.set_crossfade_opacity(value / 100.0)

    def _on_reference_changed(self, idx: int) -> None:
        self.canvas.set_reference_index(idx)

    def _on_scale_changed(self, factor: float, sd: float) -> None:
        if factor != factor or factor is None:  # NaN check
            self.scale_label.setText("Scale: n/a")
        elif sd:
            self.scale_label.setText(f"Scale: {factor:.2f} px/cm (±{sd:.2f})")
        else:
            self.scale_label.setText(f"Scale: {factor:.2f} px/cm")

    def _update_landmark_table(self) -> None:
        """Refresh the landmark table to show up to 11 landmarks."""
        row_count = 11
        self.landmark_table.setRowCount(row_count)
        for row in range(row_count):
            if row < len(self.canvas.landmarks):
                lm = self.canvas.landmarks[row]
                x_text = f"{lm.x:.3f}"
                y_text = f"{lm.y:.3f}"
                checked = lm.visible
            else:
                x_text = ""
                y_text = ""
                checked = False
            x_item = QtWidgets.QTableWidgetItem(x_text)
            y_item = QtWidgets.QTableWidgetItem(y_text)
            chk = QtWidgets.QCheckBox()
            chk.setChecked(checked)
            chk.stateChanged.connect(
                lambda state, r=row: self._on_visibility_changed(r, state)
            )
            self.landmark_table.setItem(row, 0, x_item)
            self.landmark_table.setItem(row, 1, y_item)
            self.landmark_table.setCellWidget(row, 2, chk)

    def _on_visibility_changed(self, idx: int, state: int) -> None:
        if 0 <= idx < len(self.canvas.landmarks):
            lm = self.canvas.landmarks[idx]
            self.canvas.landmarks[idx] = Landmark(lm.x, lm.y, state == QtCore.Qt.Checked)
            self.canvas.update()

    # helpers ------------------------------------------------------------
    def _load_current(self) -> None:
        if self.current_index is None or not self.image_paths:
            return
        img_path = self.image_paths[self.current_index]
        image_id = img_path.stem
        img, _ = image_manager.load_and_preprocess(image_id)
        img = img.convert("RGB")
        w, h = img.size
        qimg = QtGui.QImage(img.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.canvas.load_pixmap(pixmap)
        self.canvas.set_mode("rotate")
        self.rotate_button.setChecked(True)
        self.current_id = image_id
        self.stats_label.setText("")
        annotation_files = sorted(ANNOTATION_DIR.glob(f"{image_id}*.json"))
        if annotation_files:
            res = QtWidgets.QMessageBox.question(
                self,
                "Load Annotations?",
                "Existing annotations found. Load them?",
            )
            if res == QtWidgets.QMessageBox.Yes:
                all_sets: list[list[Landmark]] = []
                first_data: dict | None = None
                for p in annotation_files:
                    with open(p, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if first_data is None:
                        first_data = data
                    lms = [
                        Landmark(
                            lm.get("x", 0.0),
                            lm.get("y", 0.0),
                            lm.get("visible", True),
                        )
                        for lm in data.get("landmarks", [])
                    ]
                    all_sets.append(lms)
                if all_sets:
                    self.canvas.landmarks = all_sets[0]
                    self.canvas.landmarks_changed.emit()
                    if first_data and first_data.get("pixels_per_cm") is not None:
                        self.canvas.pixels_per_cm = first_data["pixels_per_cm"]
                        self.canvas.pixels_per_cm_sd = first_data.get("pixels_per_cm_sd")
                        self.canvas.scale_changed.emit(
                            self.canvas.pixels_per_cm,
                            self.canvas.pixels_per_cm_sd or float("nan"),
                        )
                if len(all_sets) > 1:
                    diffs: list[tuple[float, float]] = []
                    for idx in range(len(all_sets[0])):
                        coords = [
                            (s[idx].x * w, s[idx].y * h)
                            for s in all_sets
                            if idx < len(s)
                        ]
                        if len(coords) < 2:
                            diffs.append((float("nan"), float("nan")))
                            continue
                        dists = []
                        for i in range(len(coords)):
                            for j in range(i + 1, len(coords)):
                                dx = coords[i][0] - coords[j][0]
                                dy = coords[i][1] - coords[j][1]
                                dists.append(math.hypot(dx, dy))
                        if not dists:
                            diffs.append((float("nan"), float("nan")))
                            continue
                        mean = sum(dists) / len(dists)
                        sd = 0.0
                        if len(dists) > 1:
                            var = sum((d - mean) ** 2 for d in dists) / (len(dists) - 1)
                            sd = math.sqrt(var)
                        diffs.append((mean, sd))
                    summary_lines = [
                        f"LM {i+1}: {m:.2f} ± {s:.2f} px" if m == m else f"LM {i+1}: n/a"
                        for i, (m, s) in enumerate(diffs)
                    ]
                    self.stats_label.setText("; ".join(summary_lines))
            else:
                self.canvas.landmarks = []
                self.canvas.landmarks_changed.emit()
        self._update_landmark_table()

    def _update_buttons(self) -> None:
        if not self.image_paths or self.current_index is None:
            self.next_button.setEnabled(False)
            self.back_button.setEnabled(False)
            return
        self.back_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_paths) - 1)


def run() -> None:
    app = QtWidgets.QApplication([])
    win = AnnotatorApp()
    win.showMaximized()
    app.exec_()


if __name__ == "__main__":
    run()
