"""Simple PyQt based GUI for landmark annotation."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from FIT_python.soft_config import SOFT_CONFIG
from FIT_python.gui_annotator import image_manager
from FIT_python.gui_annotator.annotation_canvas import AnnotationCanvas, Landmark

CFG = SOFT_CONFIG.get("gui_annotator", {})
RAW_DIR = Path(CFG.get("raw_image_dir", "data/raw/images"))
ANNOTATION_DIR = Path(CFG.get("annotation_dir", "data/processed/annotations"))


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
        self.scale_button = QtWidgets.QPushButton("Scale Mode")
        self.scale_button.setCheckable(True)
        self.scale_label = QtWidgets.QLabel("Scale: n/a")
        self.landmark_table = QtWidgets.QTableWidget(0, 3)
        self.landmark_table.setHorizontalHeaderLabels(["X", "Y", "Visible"])
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
        scale_controls = QtWidgets.QHBoxLayout()
        scale_controls.addWidget(self.scale_button)
        scale_controls.addWidget(self.scale_label)
        scale_controls.addStretch()
        layout.addLayout(scale_controls)
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
        self.scale_button.toggled.connect(self.canvas.set_scale_mode)
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
        out_path = ANNOTATION_DIR / f"{self.current_id}.json"
        data = {
            "image_path": str(RAW_DIR / f"{self.current_id}.jpg"),
            "landmarks": [
                {"x": lm.x, "y": lm.y, "visible": lm.visible}
                for lm in self.canvas.landmarks
            ],
        }
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
        self.landmark_table.setRowCount(len(self.canvas.landmarks))
        for row, lm in enumerate(self.canvas.landmarks):
            x_item = QtWidgets.QTableWidgetItem(f"{lm.x:.3f}")
            y_item = QtWidgets.QTableWidgetItem(f"{lm.y:.3f}")
            chk = QtWidgets.QCheckBox()
            chk.setChecked(lm.visible)
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
        self.current_id = image_id
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
