"""Simple PyQt based GUI for landmark annotation."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from FIT_python.soft_config import SOFT_CONFIG
from . import image_manager
from .annotation_canvas import AnnotationCanvas

CFG = SOFT_CONFIG.get("gui_annotator", {})
RAW_DIR = Path(CFG.get("raw_image_dir", "data/raw/images"))
ANNOTATION_DIR = Path(CFG.get("annotation_dir", "data/processed/annotations"))


class AnnotatorApp(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FIT Image Annotator")
        self.canvas = AnnotationCanvas()
        self.open_button = QtWidgets.QPushButton("Open Image")
        self.save_button = QtWidgets.QPushButton("Save Annotations")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.save_button)
        layout.addLayout(controls)
        self.current_id: str | None = None
        self.open_button.clicked.connect(self.on_open)
        self.save_button.clicked.connect(self.on_save)

    # slots --------------------------------------------------------------
    def on_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", str(RAW_DIR))
        if not path:
            return
        image_id = Path(path).stem
        img, _ = image_manager.load_and_preprocess(image_id)
        img = img.convert("RGB")
        w, h = img.size
        qimg = QtGui.QImage(img.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.canvas.load_pixmap(pixmap)
        self.current_id = image_id

    def on_save(self) -> None:
        if not self.current_id:
            return
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ANNOTATION_DIR / f"{self.current_id}.json"
        data = {
            "image_path": str(RAW_DIR / f"{self.current_id}.jpg"),
            "landmarks": self.canvas.landmarks,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        QtWidgets.QMessageBox.information(self, "Saved", f"Annotations written to {out_path}")


def run() -> None:
    app = QtWidgets.QApplication([])
    win = AnnotatorApp()
    win.show()
    app.exec_()


if __name__ == "__main__":
    run()
