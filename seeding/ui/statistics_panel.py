"""Compact statistics panel for the simplified Seeding GUI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtWidgets import QGridLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from seeding.models import OriginalImage


@dataclass(frozen=True)
class StatisticsSummary:
    pages_count: int = 0
    objects_count: int = 0
    seedlings_count: int = 0
    inflorescences_count: int = 0
    stems_count: int = 0
    roots_count: int = 0
    other_parts_count: int = 0
    avg_confidence: float = 0.0
    min_area: int = 0
    max_area: int = 0
    histogram: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)


class StatisticsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("statisticsPanel")
        self._summary = StatisticsSummary()
        self._build_ui()
        self.set_summary(self._summary)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        self.pages_label = QLabel("")
        self.objects_label = QLabel("")
        self.avg_conf_label = QLabel("")
        self.seedlings_label = QLabel("")
        self.inflorescences_label = QLabel("")
        self.stems_label = QLabel("")
        self.roots_label = QLabel("")
        self.other_parts_label = QLabel("")
        self.min_area_label = QLabel("")
        self.max_area_label = QLabel("")

        for label in (
            self.pages_label,
            self.objects_label,
            self.avg_conf_label,
            self.seedlings_label,
            self.inflorescences_label,
            self.stems_label,
            self.roots_label,
            self.other_parts_label,
            self.min_area_label,
            self.max_area_label,
        ):
            label.setWordWrap(True)
            label.setStyleSheet("color: #f3f7fb;")

        grid.addWidget(self.pages_label, 0, 0)
        grid.addWidget(self.objects_label, 0, 1)
        grid.addWidget(self.avg_conf_label, 1, 0, 1, 2)
        grid.addWidget(self.seedlings_label, 2, 0)
        grid.addWidget(self.inflorescences_label, 2, 1)
        grid.addWidget(self.stems_label, 3, 0)
        grid.addWidget(self.roots_label, 3, 1)
        grid.addWidget(self.other_parts_label, 4, 0, 1, 2)
        grid.addWidget(self.min_area_label, 5, 0)
        grid.addWidget(self.max_area_label, 5, 1)
        layout.addLayout(grid)

        self.hist_bars: list[QProgressBar] = []
        for label in ("0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"):
            row = QGridLayout()
            caption = QLabel(label)
            caption.setStyleSheet("color: #f3f7fb;")
            row.addWidget(caption, 0, 0)
            bar = QProgressBar(self)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%v")
            bar.setStyleSheet("color: #f3f7fb;")
            row.addWidget(bar, 0, 1)
            layout.addLayout(row)
            self.hist_bars.append(bar)
        layout.addStretch()

    @staticmethod
    def build_summary(data: OriginalImage) -> StatisticsSummary:
        pages_count = len(data.images)
        objects: list = []
        if data.class_object_image:
            for page_objects in data.class_object_image:
                objects.extend(page_objects)

        objects_count = len(objects)
        if objects_count == 0:
            return StatisticsSummary(pages_count=pages_count)

        inflorescences_count = 0
        stems_count = 0
        roots_count = 0
        other_parts_count = 0
        confidences = [float(obj.confidence) for obj in objects]
        areas: list[int] = []

        for obj in objects:
            if obj.bbox:
                x1, y1, x2, y2 = obj.bbox
                areas.append(max(0, int(x2 - x1)) * max(0, int(y2 - y1)))
            for part in obj.image_all_class or []:
                part_name = StatisticsPanel._normalize_part_name(part.class_name)
                if part_name == "inflorescence":
                    inflorescences_count += 1
                elif part_name == "stem":
                    stems_count += 1
                elif part_name == "root":
                    roots_count += 1
                else:
                    other_parts_count += 1

        hist, _ = np.histogram(
            np.array(confidences, dtype=np.float32),
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.00001],
        )
        return StatisticsSummary(
            pages_count=pages_count,
            objects_count=objects_count,
            seedlings_count=objects_count,
            inflorescences_count=inflorescences_count,
            stems_count=stems_count,
            roots_count=roots_count,
            other_parts_count=other_parts_count,
            avg_confidence=float(np.mean(confidences)),
            min_area=min(areas) if areas else 0,
            max_area=max(areas) if areas else 0,
            histogram=tuple(int(v) for v in hist.tolist()),
        )

    @staticmethod
    def _normalize_part_name(name: str | None) -> str:
        value = (name or "").strip().lower()
        if value in {"соцветие", "цветок", "flower", "inflorescence"}:
            return "inflorescence"
        if value in {"стебель", "stem"}:
            return "stem"
        if value in {"корень", "root"}:
            return "root"
        return "other"

    def set_summary(self, summary: StatisticsSummary) -> None:
        self._summary = summary
        self.pages_label.setText(f"Страниц: {summary.pages_count}")
        self.objects_label.setText(f"Объектов: {summary.objects_count}")
        self.avg_conf_label.setText(
            f"Средняя уверенность: {summary.avg_confidence:.3f}"
        )
        self.seedlings_label.setText(f"Сеянцев: {summary.seedlings_count}")
        self.inflorescences_label.setText(
            f"Соцветий: {summary.inflorescences_count}"
        )
        self.stems_label.setText(f"Стеблей: {summary.stems_count}")
        self.roots_label.setText(f"Корней: {summary.roots_count}")
        self.other_parts_label.setText(
            f"Прочих частей: {summary.other_parts_count}"
        )
        self.min_area_label.setText(f"Мин. площадь: {summary.min_area}")
        self.max_area_label.setText(f"Макс. площадь: {summary.max_area}")

        bar_max = max(1, max(summary.histogram) if summary.histogram else 0)
        for idx, bar in enumerate(self.hist_bars):
            bar.setRange(0, bar_max)
            bar.setValue(summary.histogram[idx] if idx < len(summary.histogram) else 0)
