# Архитектура проекта Seeding

Документ описывает внутреннюю структуру приложения, разделение ответственности между модулями, поток данных и рекомендации по улучшению кодовой базы.

---

## Обзор

**Seeding** — локальное десктопное приложение. Архитектура построена по классической многоуровневой схеме (Layered Architecture) без серверной части:

```
┌─────────────────────────────────────────────────┐
│                   UI Layer                       │
│   main_window · login_dialog · tree_widget       │
│   thumbnails_panel · statistics_panel · bbox_item│
└────────────────────┬────────────────────────────┘
                     │  вызовы функций
┌────────────────────▼────────────────────────────┐
│               Service Layer                      │
│        services.py · user_service.py             │
│        report.py · auth.py                       │
└──────────┬──────────────────┬───────────────────┘
           │                  │
┌──────────▼──────┐  ┌────────▼──────────────────┐
│  Inference Layer│  │       Data Layer            │
│  inference.py   │  │   database.py · schema.sql  │
│  (Ultralytics)  │  │   seeding.sqlite3           │
└─────────────────┘  └────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────┐
│               Domain Models                      │
│   models.py: OriginalImage · ObjectImage         │
│   AllClassImage · AppState · SelectionPayload    │
└─────────────────────────────────────────────────┘
```

---

## Модули и ответственности

| Модуль | Слой | Ответственность |
|---|---|---|
| `main.py` | Entry point | Инициализация приложения, `SessionController`, цикл сессии |
| `auth.py` | Service | PBKDF2-хэширование паролей, `AuthUser` dataclass |
| `user_service.py` | Service | Валидация, CRUD пользователей, запись логов, иерархия исключений |
| `database.py` | Data | Низкоуровневый CRUD SQLite, управление подключениями |
| `services.py` | Service | Детекция, классификация, поворот изображений, запуск генерации отчёта |
| `inference.py` | Inference | Абстракция над Ultralytics YOLO, нормализация результатов |
| `report.py` | Service | Генерация PDF с аннотированными изображениями и таблицами |
| `models.py` | Domain | Доменные dataclass-модели данных |
| `config.py` | Config | Глобальные константы, пороги, пути к моделям |
| `manage_users.py` | CLI | Управление пользователями из терминала |
| `ui/main_window.py` | UI | Главное окно: холст, файловая панель, меню, диспетчер событий |
| `ui/login_dialog.py` | UI | Безрамочный диалог авторизации |
| `ui/tree_widget.py` | UI | Дерево страниц / объектов / частей |
| `ui/thumbnails_panel.py` | UI | Сетка миниатюр с выбором страницы |
| `ui/statistics_panel.py` | UI | Агрегированная статистика и гистограмма уверенности |
| `ui/bbox_item.py` | UI | Интерактивная ограничивающая рамка на холсте |
| `utils/geometry.py` | Util | NMS, поворот bbox и полигонов, аффинные преобразования |

---

## Поток данных

```
Пользователь открывает файл
        │
        ▼
main_window._append_page()
  → OriginalImage.images[], source_files[]
        │
        ▼
main_window.find_seedlings()
  → services.run_detection(app_state, backend)
      → inference.TorchYoloBackend.run()
      → normalize_yolo_results()           → InferenceResult
      → geometry.simple_nms()              → фильтрация
      → services.build_detected_objects()  → [ObjectImage]
  → OriginalImage.class_object_image[page]
        │
        ▼
main_window.classify_selection()
  → services.run_classification_for_selection(app_state, backend)
      → inference.TorchYoloBackend.run()   → InferenceResult
      → services.build_classified_parts()  → [AllClassImage]
  → ObjectImage.image_all_class[]
        │
        ▼
main_window.generate_report()
  → services.generate_report(app_state)
      → report.create_pdf_report()
          → ReportLab Story → PDF-файл
```

---

## База данных

Схема определена в `seeding/data/schema.sql`.

### Таблица `users`

| Столбец | Тип | Описание |
|---|---|---|
| `id` | INTEGER PK | Автоинкремент |
| `login` | TEXT UNIQUE NOT NULL | Логин в нижнем регистре |
| `password_hash` | TEXT NOT NULL | `pbkdf2_sha256$iters$salt_hex$key_hex` |
| `created_at` | TEXT | ISO-8601 timestamp |
| `updated_at` | TEXT | ISO-8601 timestamp |

### Таблица `user_logs`

| Столбец | Тип | Описание |
|---|---|---|
| `id` | INTEGER PK | Автоинкремент |
| `user_id` | INTEGER FK | Ссылка на `users.id` (CASCADE DELETE) |
| `action` | TEXT NOT NULL | Тип действия (`login`, `detect_page` и т.д.) |
| `details` | TEXT | Дополнительный контекст (опционально) |
| `created_at` | TEXT | ISO-8601 timestamp |

Индекс `idx_user_logs_user_id_created_at` ускоряет выборку логов конкретного пользователя.

---

## Модели YOLO

| Файл | Размер | Назначение |
|---|---|---|
| `models/bestDetectNew.pt` | 6.3 МБ | **Детекция сеянцев** (текущий дефолт): классы `seeding`, `seedling`, `cedr`, `pinus` |
| `models/bestKlassSeg.pt` | 54.8 МБ | **Классификация частей** (текущий дефолт): корень, стебель, соцветие и др. с масками |
| `models/bestKlassSegNew.pt` | 54.9 МБ | Обновлённая версия классификатора |
| `models/bestCropNew.pt` | 103.9 МБ | Кропирование объектов (экспериментальный) |
| `models/yolo26n.pt` | 5.5 МБ | Nano-вариант для быстрых экспериментов |
| `models/yolov8l.pt` | 87.8 МБ | YOLOv8-large для сравнения метрик |

Путь к активным моделям задаётся через `YOLO_WEIGHTS_PATH` / `YOLO_CLASSIFY_WEIGHTS_PATH` или в `seeding/config.py`.

---

## Оценка структуры: сильные стороны

### Безопасность аутентификации
Пароли хэшируются PBKDF2-HMAC-SHA256 с 120 000 итерациями и случайной солью. Проверка выполняется через `hmac.compare_digest()` — защита от timing-атак. Пароль никогда не логируется и не хранится в открытом виде.

### Разделение ответственности
UI не содержит бизнес-логики: всё вынесено в `services.py`. Слой `user_service` изолирует UI от деталей SQLite через иерархию доменных исключений.

### Абстракция инференса
Класс `InferenceBackend` (ABC) в `inference.py` позволяет подменить реализацию модели без изменений в `services.py` или UI. Нормализованный формат `InferenceResult` / `InferenceBox` изолирует код от деталей Ultralytics API.

### Конфигурация через переменные окружения
Пути к моделям и БД переопределяются без изменения кода. Это позволяет использовать разные конфигурации в тестах, разработке и продакшне.

### Оптимизации геометрии
`geometry.py` разделяет быстрый путь (повороты на 90° через `np.rot90` / перестановку координат) и общий случай (аффинные преобразования через OpenCV). Плавающая точка не накапливается при кратных 90°-поворотах.

---

## Оценка структуры: рекомендации по улучшению

### 1. Разбить `main_window.py` (критично)

**Проблема:** файл превышает 2000 строк и совмещает ответственности холста, тулбара, файловой панели и диспетчера событий.

**Рекомендация:** разделить на 4 модуля:

```
ui/
├── window.py           # QMainWindow: компоновка, жизненный цикл
├── canvas_view.py      # QGraphicsView: отрисовка, масштаб, измерение
├── file_panel.py       # Список файлов, добавление страниц
└── toolbar_actions.py  # Toolbar, меню, горячие клавиши
```

---

### 2. Убрать или наполнить `seeding/processing/`

**Проблема:** пустая директория вводит в заблуждение — нет ни кода, ни `__init__.py`.

**Рекомендация:** удалить директорию или перенести в неё часть логики из `services.py` (обработка изображений, поворот).

---

### 3. Перенести инференс в фоновый поток

**Проблема:** YOLO-инференс выполняется в главном потоке PyQt5. При обработке многостраничных PDF интерфейс зависает.

**Рекомендация:** вынести `find_seedlings()` и `classify_selection()` в `QThread` или `QRunnable` с прогресс-баром.

```python
class DetectionWorker(QThread):
    progress = pyqtSignal(int, int)      # (текущая, всего)
    finished = pyqtSignal(list)          # результаты

    def run(self):
        for i, page in enumerate(pages):
            result = run_detection(...)
            self.progress.emit(i + 1, len(pages))
        self.finished.emit(results)
```

---

### 4. Кешировать флаг инициализации хранилища

**Проблема:** `initialize_user_storage()` вызывается при каждой операции с базой данных, что приводит к лишнему чтению схемы.

**Рекомендация:** кешировать результат в модульной переменной или использовать одноразовую инициализацию при запуске.

```python
_storage_ready: bool = False

def _ensure_storage() -> None:
    global _storage_ready
    if not _storage_ready:
        initialize_database()
        _storage_ready = True
```

---

### 5. Добавить тесты для непокрытых модулей

| Модуль | Приоритет | Что тестировать |
|---|---|---|
| `utils/geometry.py` | Высокий | NMS с перекрытием, поворот полигонов при накоплении |
| `services.py` | Высокий | `run_detection` с mock-бэкендом, `rotate_page` с проверкой bbox |
| `report.py` | Средний | Генерация PDF не бросает исключений, файл создаётся |
| `manage_users.py` | Средний | Все субкоманды CLI через `subprocess` или `argparse` |

---

### 6. Вынести датасеты и модели из репозитория

**Проблема:** `dataset/` (16+ версий, бинарные файлы) и `models/` (до 155 МБ на файл) хранятся в Git, что делает клонирование медленным.

**Рекомендации:**
- Модели → [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) или [DVC](https://dvc.org/)
- Датасеты → DVC + удалённое хранилище (S3, GDrive)
- Добавить `models/` и `dataset/` в `.gitignore` (кроме `README.md`)

---

### 7. Добавить undo/redo для трансформаций

**Проблема:** поворот изображения необратим без перезагрузки файла.

**Рекомендация:** реализовать стек команд (Command Pattern):

```python
@dataclass
class RotateCommand:
    page_index: int
    rotation_k: int

    def execute(self, state: AppState) -> None: ...
    def undo(self, state: AppState) -> None: ...
```

---

## Дерево файлов с аннотациями

```text
seeding/
├── main.py
│   └── SessionController     # login → window → logout цикл
├── auth.py
│   ├── AuthUser              # frozen dataclass: id, login
│   ├── hash_password()       # PBKDF2-HMAC-SHA256, 120k итераций
│   └── verify_password()     # hmac.compare_digest
├── config.py                 # Все магические числа в одном месте
├── models.py
│   ├── OriginalImage         # Контейнер проекта (страницы + объекты)
│   ├── ObjectImage           # Обнаруженный сеянец + части
│   ├── AllClassImage         # Классифицированная часть растения
│   ├── AppState              # Текущее состояние сессии
│   └── SelectionPayload      # Метаданные выбранного элемента UI
├── inference.py
│   ├── InferenceBackend      # ABC: run(image) → InferenceResult
│   ├── TorchYoloBackend      # Ultralytics YOLO реализация
│   ├── InferenceResult       # names: dict[int,str], boxes: list[InferenceBox]
│   ├── InferenceBox          # cls, conf, bbox, mask_polygon
│   └── normalize_yolo_results()
├── services.py               # Чистые функции: принимают AppState, возвращают изменения
│   ├── run_detection()
│   ├── run_classification_for_selection()
│   ├── rotate_selection()
│   └── generate_report()
├── report.py
│   └── create_pdf_report()   # ReportLab Story с аннотированными изображениями
├── database.py               # Все SQL-запросы; оборачивает ошибки в DatabaseError
├── user_service.py           # Валидация + CRUD + логирование; исключения-домены
├── manage_users.py           # argparse CLI: create/list/set-password/delete
└── utils/
    └── geometry.py
        ├── simple_nms()              # Жадный NMS по IoU
        ├── rotate_bbox()             # 90° шаги без потери точности
        ├── rotate_polygon_points()   # Быстрый путь для 90°, аффинный для произвольных
        └── rotate_image_and_boxes()  # np.rot90 или cv2.warpAffine
```
