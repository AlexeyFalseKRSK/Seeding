"""Диалог временной авторизации перед входом в приложение."""

from __future__ import annotations

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

from seeding.auth import (
    TEST_USER,
    TEST_USER_PASSWORD,
    AuthUser,
    authenticate,
)


class LoginDialog(QDialog):
    """Показывает минимальный экран входа перед запуском главного окна."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Инициализирует диалог и создаёт элементы формы входа."""

        super().__init__(parent)
        self._authenticated_user: AuthUser | None = None
        self._drag_offset = QPoint()
        self._drag_active = False
        self.setObjectName("loginDialog")
        self.setWindowTitle("Seeding - Вход")
        self.setModal(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedWidth(420)
        self._build_ui()
        self.adjustSize()

    @property
    def authenticated_user(self) -> AuthUser | None:
        """Возвращает успешно аутентифицированного пользователя."""

        return self._authenticated_user

    def _build_ui(self) -> None:
        """Создаёт минимальный интерфейс диалога входа."""

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

        subtitle = QLabel("Тестовый вход в приложение", card)
        subtitle.setObjectName("panelSubTitle")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)

        chip = QLabel(TEST_USER.display_name, card)
        chip.setObjectName("metricChip")
        chip.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(chip, alignment=Qt.AlignCenter)

        note = QLabel(
            f"Тестовый доступ: {TEST_USER.email} / {TEST_USER_PASSWORD}",
            card,
        )
        note.setObjectName("panelHint")
        note.setAlignment(Qt.AlignCenter)
        note.setWordWrap(True)
        card_layout.addWidget(note)

        email_label = QLabel("E-mail", card)
        email_label.setObjectName("panelSubTitle")
        card_layout.addWidget(email_label)

        self.email_input = QLineEdit(card)
        self.email_input.setPlaceholderText("user@user.com")
        self.email_input.setClearButtonEnabled(True)
        self.email_input.textChanged.connect(self._clear_error)
        self.email_input.returnPressed.connect(self._focus_password_input)
        card_layout.addWidget(self.email_input)

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

        self.email_input.setFocus()

    def _focus_password_input(self) -> None:
        """Переводит фокус на поле пароля после ввода e-mail."""

        self.password_input.setFocus(Qt.TabFocusReason)
        self.password_input.selectAll()

    def _clear_error(self) -> None:
        """Скрывает сообщение об ошибке, когда пользователь меняет ввод."""

        self.error_label.clear()
        self.error_label.hide()

    def _attempt_login(self) -> None:
        """Пытается аутентифицировать пользователя по введённым данным."""

        user = authenticate(self.email_input.text(), self.password_input.text())
        if user is None:
            self._authenticated_user = None
            self.password_input.clear()
            self.error_label.setText("Неверный логин или пароль.")
            self.error_label.show()
            self.password_input.setFocus(Qt.TabFocusReason)
            return

        self._authenticated_user = user
        self.accept()

    def mousePressEvent(self, event) -> None:
        """Позволяет перетаскивать безрамочное окно мышью."""

        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Перемещает диалог вслед за курсором во время перетаскивания."""

        if self._drag_active and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Завершает перетаскивание окна после отпускания кнопки мыши."""

        if event.button() == Qt.LeftButton:
            self._drag_active = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
