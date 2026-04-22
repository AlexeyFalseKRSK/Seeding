# Changelog

Все значимые изменения проекта Seeding фиксируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).

---

## [Unreleased]

### Added
- `seeding/ui/main_window.py` — удаление аннотаций из дерева слоев: горячие клавиши `Delete`/`Backspace` и контекстное меню `Удалить` для узлов `seeding` и `class`.
- `tests/test_main_window_resilience.py` — интеграционные сценарии:
  - ручное добавление части с проверкой `local bbox` и `mask_bitmap` в координатах seed crop;
  - удаление части и сеянца через единый обработчик удаления;
  - предсказуемое поведение режима `+ Бокс` при отмене диалога и при попытке добавить часть без выбранного сеянца.
- `tests/test_add_box_dialog.py` — проверка явного визуального выбора класса в диалоге ручного добавления.

### Changed
- `seeding/ui/main_window.py` — режим `Ред. боксов` включен и синхронизирован с `_interaction_mode`.
- `seeding/ui/main_window.py` — ручное добавление части (`root/stem/inflorescence`) теперь сохраняет `AllClassImage.bbox` в локальных координатах seed crop.
- `seeding/ui/main_window.py` — `mask_bitmap` для ручных частей строится в размере seed crop (единая система координат с auto-классификацией).
- `seeding/ui/main_window.py` — detail panel корректно показывает crop/маску части для локального `bbox`.
- `seeding/ui/main_window.py` — рендер маски части поддерживает legacy-случай bitmap размера bbox (автоматическая проекция в размер текущего crop).
- `seeding/ui/add_box_dialog.py` — выбор класса сделан явным: эксклюзивное состояние, заметная подсветка активной кнопки и подпись текущего выбора.
- `seeding/ui/main_window.py` — поток `draw -> choose class -> add object` больше не содержит тихих отказов: ошибки и отмены показываются через status bar, а режим `+ Бокс` выключается только после успешного добавления.

---

## [1.8.0] - 2026-04-20

### Added
- `seeding/mask_refiner.py` — модуль попиксельного уточнения сегментационных масок:
  - `polygon_to_bitmap` — растеризация YOLO-полигона в бинарную маску
  - `refine_mask_bitmap` — удаление фоновых пикселей (белая бумага) внутри грубой маски через адаптивный порог: Otsu при сбалансированной гистограмме, фиксированный порог `_PAPER_THRESHOLD=210` при разреженном объекте (тонкие корни, хвоинки); морфологическое закрытие + фильтр мелких компонент
  - `bitmap_to_polygon` — наибольший контур из bitmap обратно в полигон (совместимость)
  - `rotate_bitmap` — ротация bitmap-маски вместе с кропом
- `AllClassImage.mask_bitmap` — новое поле `np.ndarray | None`: попиксельная бинарная маска в координатах crop-изображения родительского сеянца; хранится рядом с полигоном для точного рендеринга и расчёта площади

### Changed
- `seeding/services.py` → `build_classified_parts`: после YOLO-классификации строится bitmap через `polygon_to_bitmap` → `refine_mask_bitmap`; полигон обновляется как наибольший контур уточнённой маски
- `seeding/services.py` → `rotate_crop`: при повороте кропа вращается и `mask_bitmap` через `rotate_bitmap`
- `seeding/ui/main_window.py` → `_add_part_mask_item`: приоритетный рендер через `QGraphicsPixmapItem` (RGBA bitmap-overlay + контуры всех компонент связности); fallback на полигон для данных без bitmap
- `tests/test_main_window_resilience.py` — проверка формы `mask_polygon` смягчена: вместо `shape == (4, 2)` проверяется `ndim == 2`, `shape[1] == 2`, `shape[0] >= 3` (полигон теперь извлекается из bitmap и может иметь другое число вершин)

---

## [1.7.0] - 2026-04-19

### Added
- `seeding/database.py` — низкоуровневый CRUD-слой SQLite: пользователи, логи, управление подключениями
- `seeding/user_service.py` — сервисный слой пользователей: валидация логина/пароля, иерархия исключений (`StorageUnavailableError`, `DuplicateUserError`, `UserNotFoundError`)
- `seeding/manage_users.py` — CLI для управления учётными записями (`seeding-users create/list/set-password/delete`)
- `seeding/data/schema.sql` — схема SQLite-базы: таблицы `users`, `user_logs`, индекс по `user_id + created_at`
- `seeding/data/seeding.sqlite3` — локальная база данных пользователей (создаётся при первом запуске)
- `dataset/species_split/` — новый датасет для разделения сеянцев по видам
- `tools/build_species_detection_dataset.py` — скрипт подготовки многоклассового датасета из Roboflow-формата
- `models/yolo26n.pt` — nano-версия модели YOLO для экспериментов

### Changed
- `seeding/auth.py` — интегрирован с новым storage-слоем; `AuthUser` остаётся независимым доменным объектом
- `seeding/main.py` — `SessionController` использует `initialize_user_storage()` из `user_service`
- `seeding/ui/login_dialog.py` — статус хранилища (`_refresh_storage_state`) подключён к новому сервисному слою
- `seeding/ui/main_window.py` — логирование действий (`_log_action`) переведено на `record_user_action()` из `user_service`
- `tests/test_auth.py` — тесты расширены: проверка пути к БД, схема таблиц, диалог авторизации с изолированной тестовой БД
- `tests/test_inference.py` — обновлены стабы под актуальный API `normalize_yolo_results`
- `tests/test_main_window_resilience.py` — монопатчинг `SEEDING_DB_PATH` для изоляции тестов
- `README.md` — полностью переработан: структура, конфигурация, таблица покрытия тестов, описание CLI

### Removed
- Устаревшие временные файлы из корня проекта (`~$*.docx`, `~WRL*.tmp`, `Диплом.docx`)
- Документы перемещены из корня в `docs/requirements/`, `docs/standards/`, `docs/thesis/`

---

## [1.6.0] - 2026-04-19

### Added
- Базовая авторизация пользователей при запуске приложения
- `seeding/ui/login_dialog.py` — безрамочный диалог входа с перетаскиванием, статусными сообщениями и inline-ошибками
- `seeding/auth.py` — хэширование паролей PBKDF2-HMAC-SHA256 (120 000 итераций, случайная соль), константно-временная проверка через `hmac.compare_digest`
- `tests/test_auth.py` — начальный набор тестов авторизации

### Changed
- `seeding/main.py` — добавлен `SessionController`: управление циклом логин → главное окно → выход
- `seeding/ui/main_window.py` — поддержка `current_user`, сигнал `logout_requested`, действие выхода из системы
- `seeding/resources/styles/dark.qss` — обновлён под безрамочный диалог
- `tests/test_main_window_resilience.py` — добавлены тесты logout-сигнала и drag-поведения

### Removed
- `dataset/datasetAnalisV1` — устаревший датасет удалён

---

## [1.5.0] - 2026-03-28

### Added
- Детекция видов `cedr` (кедр) и `pinus` (сосна) добавлена в основной пайплайн
- `dataset/datasetV1_species/` — датасет для детекции пород
- `tools/train_species_detection.py` — скрипт обучения детектора видов
- `models/bestDetectNew.pt` — новая модель детекции с поддержкой кедра и сосны
- `results/cedr_pinus_detect/` — метрики и графики обучения

### Changed
- `seeding/config.py` — `DETECTION_CLASS_NAMES` расширен до `("seeding", "seedling", "cedr", "pinus")`; добавлены русские отображаемые имена
- `seeding/services.py` — фильтрация по `allowed_classes` в `build_detected_objects`
- `seeding/ui/main_window.py`, `seeding/ui/statistics_panel.py`, `seeding/report.py` — обновлены под новые классы

---

## [1.4.0] - 2026-03-22

### Changed
- Основная прикладная логика выделена в `seeding/services.py` (детекция, классификация, поворот, генерация отчёта)
- Снижена связанность между UI и вычислительным слоем
- `seeding/ui/main_window.py` — вызовы переведены на функции из `services.py`
- `tests/test_main_window_resilience.py` — адаптированы под новую архитектуру

---

## [1.3.0] - 2026-03-13

### Added
- Примерное техническое задание `TZ.md` (перемещено в `docs/requirements/` в v1.7.0)

### Changed
- `README.md` — актуализирован под состояние проекта

---

## [1.2.0] - 2026-03-12

### Added
- Требования и материалы: `BUSINESS_REQUIREMENTS.md`, `BUSINESS_IMPROVEMENTS.md`, `Gost_34_602-2020.pdf`, материалы ВКР
- Результаты обучения: `coverage.xml`, `results/seeding_segmentation/`, `results/seedings_detection/`
- Поддержка масок сегментации частей сеянца (`mask_polygon` в `AllClassImage`)
- Русскоязычные docstring-описания для ключевых функций и классов

### Changed
- `seeding/config.py`, `seeding/models.py`, `seeding/services.py`, `seeding/ui/*` — рефакторинг под маски и новые классы

### Removed
- Неиспользуемые функции и методы (чистка кода)

---

## [1.1.0] - 2026-03-11

### Added
- Демонстрационные входные и выходные материалы в `Photo/`
- Датасет `dataset/datasetAnalisV1/` (удалён в v1.6.0)
- Файлы весов моделей и результаты первых экспериментов
- `.gitignore`, `.gitattributes`

---

## [1.0.0] - 2026-03-11

### Added
- Первый рабочий релиз desktop-приложения Seeding
- Ядро приложения: `main.py`, `config.py`, `services.py`, `inference.py`, `models.py`, `report.py`
- UI: `seeding/ui/main_window.py` — главное окно с холстом, деревом слоёв, миниатюрами и статистикой
- Утилиты: `seeding/utils/geometry.py` — NMS, повороты bbox и полигонов, аффинные преобразования
- Ресурсы: 11 SVG-иконок, тёмная QSS-тема
- Базовые пользовательские сценарии: открытие файлов, детекция, классификация, формирование отчёта
- Начальный набор автоматических тестов (pytest)
