# Seeding

![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue)
![GUI](https://img.shields.io/badge/GUI-PyQt5-green)
![YOLO](https://img.shields.io/badge/AI-Ultralytics%20YOLO-orange)
![Tests](https://img.shields.io/badge/tests-pytest-yellowgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Десктопное приложение Python/PyQt5 для анализа изображений посадочного материала с помощью моделей YOLO. Поддерживает детекцию сеянцев, сегментацию частей растений и формирование PDF-отчётов.

---

## Содержание

- [Возможности](#возможности)
- [Технологии](#технологии)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация](#конфигурация)
- [Рабочий процесс](#рабочий-процесс)
- [Горячие клавиши](#горячие-клавиши)
- [Управление пользователями](#управление-пользователями)
- [Архитектура](#архитектура)
- [Тестирование](#тестирование)
- [Ограничения](#ограничения)
- [Авторы](#авторы)

---

## Возможности

| Функция | Описание |
|---------|---------|
| Входные форматы | PNG, JPG, JPEG, BMP, TIFF, PDF |
| Детекция | YOLO — обнаружение сеянцев на странице или всём документе |
| Классификация | YOLO Segmentation — корень, стебель, соцветие с масками |
| Редактирование | Интерактивный ресайз bbox и масок на холсте |
| Поворот | Страниц и кропов с автоматическим пересчётом координат |
| Калибровка | Инструмент измерения (пикселей/мм) |
| Отчёт | PDF с аннотированными изображениями и таблицами |
| Пользователи | Локальная БД SQLite, PBKDF2-HMAC-SHA256, аудит-лог |
| CLI | Управление учётными записями через `seeding-users` |

---

## Технологии

| Компонент | Библиотека |
|-----------|-----------|
| GUI | PyQt5 >= 5.15 |
| Детекция и сегментация | Ultralytics YOLO >= 8.0 |
| Обработка изображений | OpenCV >= 4.7, Pillow >= 9.4 |
| Работа с PDF | PyMuPDF >= 1.22 |
| Генерация отчётов | ReportLab >= 3.6 |
| Вычисления | NumPy >= 1.23 |
| База данных | SQLite3 (встроен) |

---

## Быстрый старт

**Требования:** Python 3.10+, файлы весов `.pt` (см. [Конфигурация](#конфигурация))

```bash
# 1. Клонировать
git clone https://github.com/AlexeyFalseKRSK/Seeding.git
cd Seeding

# 2. Виртуальное окружение
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Установить зависимости
pip install -e .

# 4. Создать пользователя
python -m seeding.manage_users create admin

# 5. Запустить
seeding
```

---

## Конфигурация

### Пути к моделям

| Параметр | Аргумент CLI | Переменная окружения | Значение по умолчанию |
|---------|-------------|---------------------|----------------------|
| Модель детекции | `--weights` | `YOLO_WEIGHTS_PATH` | `models/bestDetectNew.pt` |
| Модель классификации | `--classify-weights` | `YOLO_CLASSIFY_WEIGHTS_PATH` | `models/bestKlassSegFlip180.pt` |
| База данных | — | `SEEDING_DB_PATH` | `seeding/data/seeding.sqlite3` |

```bash
# С явным указанием моделей
seeding --weights models/bestDetectNew.pt --classify-weights models/bestKlassSegFlip180.pt

# Через env
YOLO_WEIGHTS_PATH=models/bestDetectNew.pt seeding
SEEDING_DB_PATH=/data/mydb.sqlite3 seeding
```

### Модели YOLO

| Файл | Назначение |
|------|-----------|
| `models/bestDetectNew.pt` | Детекция сеянцев (дефолт) |
| `models/bestKlassSegFlip180.pt` | Классификация частей + маски (дефолт) |
| `models/bestKlassSeg.pt` | Альтернативный классификатор |

> Файлы весов не хранятся в репозитории (`.gitignore`). Получите их отдельно.
> Подробнее о локальных моделях, датасетах и правилах хранения артефактов:
> [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

### Пороговые значения (`seeding/config.py`)

| Параметр | Значение |
|---------|---------|
| Порог уверенности (confidence) | 0.25 |
| IoU для NMS | 0.40 |
| Высокая уверенность (зелёный) | >= 0.90 |
| Низкая уверенность (красный) | < 0.50 |

---

## Рабочий процесс

1. Запустить приложение и войти (`seeding`)
2. Открыть изображение или PDF (`Ctrl+O`)
3. Запустить детекцию (`Ctrl+F` — текущая страница, `Ctrl+Shift+F` — все)
4. Выбрать найденный сеянец в дереве слоёв
5. Классифицировать части (`Ctrl+C`)
6. При необходимости: повернуть (`Ctrl+R`), отредактировать bbox на холсте
7. Проверить статистику в правой панели
8. Сформировать PDF-отчёт (`Ctrl+P`)

---

## Горячие клавиши

| Клавиши | Действие |
|---------|---------|
| `Ctrl+O` | Открыть файл |
| `Ctrl+Shift+O` | Добавить файлы к проекту |
| `Ctrl+F` | Детекция на текущей странице |
| `Ctrl+Shift+F` | Детекция на всех страницах |
| `Ctrl+C` | Классификация выбранного объекта |
| `Ctrl+R` | Повернуть страницу/объект на 90° |
| `Ctrl+P` | Сформировать PDF-отчёт |
| `Ctrl++` / `Ctrl+-` | Масштаб + / - |
| `Ctrl+0` | Сбросить масштаб |
| Колесо мыши | Масштабирование холста |
| ПКМ + drag | Панорамирование |

---

## Управление пользователями

```bash
seeding-users create alice        # создать (пароль интерактивно)
seeding-users list                # список пользователей
seeding-users set-password alice  # сменить пароль
seeding-users delete alice        # удалить
seeding-users delete alice --yes  # удалить без подтверждения
```

**Формат логина:** 3–50 символов, `a-z 0-9 . - _`, регистронезависим.  
**Пароль:** PBKDF2-HMAC-SHA256, 120 000 итераций, случайная соль. Никогда не хранится в открытом виде.

### Действия в аудит-логе

`login` · `logout` · `detect_page` · `detect_all` · `classify` · `generate_report` · `rotate`

---

## Архитектура

Трёхслойная архитектура без серверной части:

```
UI Layer     →  seeding/ui/
Service      →  seeding/services.py, user_service.py, report.py
Data         →  seeding/database.py (SQLite), seeding/data/
```

Подробное описание: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Ключевые модули

| Файл | Роль |
|------|------|
| `main.py` | Точка входа, `SessionController` |
| `config.py` | Все константы и пороги |
| `models.py` | Доменные dataclass-ы: `AppState`, `ObjectImage`, `AllClassImage` |
| `services.py` | `run_detection()`, `run_classification_for_selection()`, `rotate_selection()` |
| `inference.py` | ABC `InferenceBackend`, `TorchYoloBackend` |
| `auth.py` | PBKDF2, `hash_password()`, `verify_password()` |
| `database.py` | SQLite CRUD, `DatabaseError` |
| `user_service.py` | Валидация, CRUD, иерархия исключений |
| `report.py` | PDF через ReportLab |
| `ui/main_window.py` | Главное окно (~2000 LOC) |
| `utils/geometry.py` | NMS, повороты bbox/полигонов |

---

## Тестирование

```bash
pytest tests/                                      # все тесты
pytest tests/ --cov=seeding --cov-report=term-missing  # с покрытием
pytest tests/test_auth.py -v                       # один файл
```

| Модуль | Статус |
|--------|--------|
| `auth.py`, `user_service.py` | покрыто |
| `inference.py` | покрыто |
| `ui/` компоненты (дерево, статистика, иконки) | покрыто |
| `ui/main_window.py` — интеграционные | покрыто |
| `utils/geometry.py` | **не покрыто** |
| `report.py` | **не покрыто** |

---

## Структура проекта

```
Seeding/
├── seeding/                # основной пакет
│   ├── ui/                 # PyQt5 компоненты
│   ├── utils/              # геометрия, пути
│   ├── resources/          # иконки SVG, QSS-стили
│   ├── data/               # schema.sql, seeding.sqlite3
│   ├── main.py             # точка входа
│   ├── config.py           # константы
│   ├── models.py           # доменные модели
│   ├── services.py         # бизнес-логика
│   ├── inference.py        # абстракция YOLO
│   ├── auth.py             # хэширование паролей
│   ├── database.py         # SQLite CRUD
│   ├── user_service.py     # сервис пользователей
│   ├── report.py           # генерация PDF
│   └── mask_refiner.py     # обработка масок
├── tests/                  # pytest
├── models/                 # веса YOLO .pt (не в git)
├── dataset/                # датасеты YOLO (не в git)
├── docs/                   # архитектура, ТЗ, диаграммы
├── scripts/                # скрипты обучения
└── CHANGELOG.md
```

---

## Ограничения

- Файлы весов `.pt` необходимы для работы — не включены в репозиторий
- Инференс YOLO в основном потоке — UI временно зависает на больших PDF
- Нет undo/redo для трансформаций изображений
- Только локальный однопользовательский режим
- Только формат `.pt` (Ultralytics/PyTorch), ONNX не поддерживается

---

## Авторы

**Валеев Алексей** — [@diristhor](https://t.me/diristhor)  
**Анна Алехина** — [@Carthago_delenda_es](https://t.me/Carthago_delenda_es)

---

История изменений: [CHANGELOG.md](CHANGELOG.md)
