"""Application settings dialog."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


@dataclass(frozen=True)
class AppSettings:
    """User-editable application settings."""

    analysis_device: str
    analysis_batch_size: int
    auto_save_enabled: bool
    report_output_dir: str


class AppSettingsDialog(QDialog):
    """Dialog for inference and application performance settings."""

    DEVICE_OPTIONS = [
        ("Авто: видеокарта, если доступна", "auto"),
        ("CPU", "cpu"),
        ("GPU 0", "0"),
        ("CUDA 0", "cuda:0"),
    ]

    def __init__(
        self,
        *,
        current_device: str,
        current_batch_size: int,
        current_auto_save_enabled: bool = True,
        current_report_output_dir: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._device_combo = QComboBox(self)
        for label, value in self.DEVICE_OPTIONS:
            self._device_combo.addItem(label, value)
        self._device_combo.setCurrentIndex(self._device_index(current_device))
        form.addRow("Сегментировать на:", self._device_combo)

        self._batch_spin = QSpinBox(self)
        self._batch_spin.setRange(1, 32)
        self._batch_spin.setValue(max(1, int(current_batch_size)))
        self._batch_spin.setToolTip(
            "Больше = быстрее на GPU, но выше расход памяти."
        )
        form.addRow("Размер batch:", self._batch_spin)

        self._auto_save_check = QCheckBox(self)
        self._auto_save_check.setChecked(bool(current_auto_save_enabled))
        self._auto_save_check.setToolTip(
            "Если включено, результаты анализа сохраняются после детекции, сегментации и создания отчета."
        )
        form.addRow("Автосохранение:", self._auto_save_check)

        report_dir_row = QHBoxLayout()
        self._report_dir_edit = QLineEdit(self)
        self._report_dir_edit.setText(str(current_report_output_dir or ""))
        self._report_dir_edit.setPlaceholderText("Рядом с исходным файлом")
        self._report_dir_edit.setClearButtonEnabled(True)
        report_dir_row.addWidget(self._report_dir_edit, 1)
        browse_button = QPushButton("Выбрать", self)
        browse_button.clicked.connect(self._select_report_dir)
        report_dir_row.addWidget(browse_button)
        form.addRow("Папка отчетов:", report_dir_row)

        layout.addLayout(form)
        note = QLabel(
            "Настройки устройства применятся к новым запускам модели. Пустая папка отчетов означает сохранение рядом с текущим исходным файлом.",
            self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_settings(self) -> AppSettings:
        """Return settings selected by the user."""
        return AppSettings(
            analysis_device=str(self._device_combo.currentData()),
            analysis_batch_size=int(self._batch_spin.value()),
            auto_save_enabled=bool(self._auto_save_check.isChecked()),
            report_output_dir=self._report_dir_edit.text().strip(),
        )

    def _device_index(self, current_device: str) -> int:
        current = str(current_device or "auto")
        for index, (_, value) in enumerate(self.DEVICE_OPTIONS):
            if value == current:
                return index
        return 0

    def _select_report_dir(self) -> None:
        """Select the default directory for generated reports."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Папка для отчетов",
            self._report_dir_edit.text().strip() or "",
        )
        if directory:
            self._report_dir_edit.setText(directory)
