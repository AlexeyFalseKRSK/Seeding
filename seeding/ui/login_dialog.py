"""Диалог входа, показываемый перед открытием главного окна приложения."""

from __future__ import annotations

import logging

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from seeding.auth import AuthUser
from seeding.user_service import (
    StorageUnavailableError,
    authenticate_user,
    has_users,
)

logger = logging.getLogger(__name__)

NO_USERS_MESSAGE = (
    "В базе пока нет пользователей. Создайте первого пользователя командой:\n"
    "python -m seeding.manage_users create <логин>"
)


class LoginDialog(QDialog):
    """Показывает минимальную форму входа перед запуском главного окна."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._authenticated_user: AuthUser | None = None
        self._drag_offset = QPoint()
        self._drag_active = False
        self._storage_error: str | None = None
        self._users_configured = False
        self.setObjectName("loginDialog")
        self.setWindowTitle("Seeding - Вход")
        self.setModal(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedWidth(440)
        self._refresh_storage_state()
        self._build_ui()
        self.adjustSize()

    @property
    def authenticated_user(self) -> AuthUser | None:
        """Возвращает пользователя после успешного входа."""

        return self._authenticated_user

    def _refresh_storage_state(self) -> None:
        """Обновляет текущее состояние доступности БД и наличия пользователей."""

        try:
            self._users_configured = has_users()
            self._storage_error = None
        except StorageUnavailableError as error:
            self._users_configured = False
            self._storage_error = str(error)

    def _status_chip_text(self) -> str:
        """Возвращает короткий статусный текст для карточки."""

        if self._storage_error:
            return "Ошибка БД"
        if self._users_configured:
            return "Авторизация"
        return "Нет пользователей"

    def _status_note_text(self) -> str:
        """Возвращает подробную статусную подсказку над формой."""

        if self._storage_error:
            return (
                "Не удалось подключиться к локальной базе данных.\n"
                f"{self._storage_error}"
            )
        if self._users_configured:
            return "Введите логин и пароль пользователя, созданного в локальной базе."
        return NO_USERS_MESSAGE

    def _build_ui(self) -> None:
        """Строит визуальную компоновку диалога входа."""

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        card = QFrame(self)
        card.setObjectName("panelCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(10)
        root_layout.addWidget(card)

        title = QLabel("Seeding", card)
        title.setObjectName("panelCardTitle")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("Авторизация в приложении", card)
        subtitle.setObjectName("panelSubTitle")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)

        self.status_chip = QLabel(self._status_chip_text(), card)
        self.status_chip.setObjectName("metricChip")
        self.status_chip.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.status_chip, alignment=Qt.AlignCenter)

        self.note_label = QLabel(self._status_note_text(), card)
        self.note_label.setObjectName("panelHint")
        self.note_label.setAlignment(Qt.AlignCenter)
        self.note_label.setWordWrap(True)
        card_layout.addWidget(self.note_label)

        login_label = QLabel("Логин", card)
        login_label.setObjectName("panelSubTitle")
        card_layout.addWidget(login_label)

        self.login_input = QLineEdit(card)
        self.login_input.setPlaceholderText("Введите логин")
        self.login_input.setClearButtonEnabled(True)
        self.login_input.textChanged.connect(self._clear_error)
        self.login_input.returnPressed.connect(self._focus_password_input)
        card_layout.addWidget(self.login_input)

        password_label = QLabel("Пароль", card)
        password_label.setObjectName("panelSubTitle")
        card_layout.addWidget(password_label)

        self.password_input = QLineEdit(card)
        self.password_input.setPlaceholderText("Введите пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.textChanged.connect(self._clear_error)
        self.password_input.returnPressed.connect(self._attempt_login)
        card_layout.addWidget(self.password_input)

        self.error_label = QLabel("", card)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #fca5a5;")
        self.error_label.hide()
        card_layout.addWidget(self.error_label)

        self.login_button = QPushButton("Войти", card)
        self.login_button.setObjectName("primaryActionButton")
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setDefault(True)
        self.login_button.clicked.connect(self._attempt_login)
        card_layout.addWidget(self.login_button)

        self.login_input.setFocus()

    def _show_error_text(self, text: str) -> None:
        """Показывает встроенное сообщение об ошибке в диалоге."""

        self.error_label.setText(text)
        self.error_label.show()

    def _focus_password_input(self) -> None:
        """Переводит фокус на поле ввода пароля."""

        self.password_input.setFocus(Qt.TabFocusReason)
        self.password_input.selectAll()

    def _clear_error(self) -> None:
        """Скрывает ошибку, когда пользователь меняет ввод."""

        self.error_label.clear()
        self.error_label.hide()

    def _attempt_login(self) -> None:
        """Проверяет учётные данные, введённые в диалоге."""

        self._refresh_storage_state()
        self.status_chip.setText(self._status_chip_text())
        self.note_label.setText(self._status_note_text())

        if self._storage_error is not None:
            self._authenticated_user = None
            self._show_error_text("База данных недоступна. Проверьте файл SQLite и схему.")
            return

        if not self._users_configured:
            self._authenticated_user = None
            self._show_error_text(NO_USERS_MESSAGE)
            return

        try:
            user = authenticate_user(
                self.login_input.text(),
                self.password_input.text(),
            )
        except StorageUnavailableError as error:
            logger.exception("Не удалось выполнить вход из-за ошибки БД")
            self._authenticated_user = None
            self._show_error_text(f"Ошибка базы данных:\n{error}")
            return

        if user is None:
            logger.warning(
                "Неуспешная попытка входа для логина '%s'",
                self.login_input.text().strip(),
            )
            self._authenticated_user = None
            self.password_input.clear()
            self._show_error_text("Неверный логин или пароль.")
            self.password_input.setFocus(Qt.TabFocusReason)
            return

        self._authenticated_user = user
        self.accept()

    def mousePressEvent(self, event) -> None:
        """Позволяет перетаскивать безрамочный диалог мышью."""

        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Перемещает диалог во время перетаскивания."""

        if self._drag_active and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Завершает перетаскивание при отпускании кнопки мыши."""

        if event.button() == Qt.LeftButton:
            self._drag_active = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
