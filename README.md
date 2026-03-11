# Seeding 1.0.0

Упрощенная дипломная версия настольного приложения для анализа изображений и PDF с сеянцами.

## Что осталось

- открытие изображений и PDF;
- детекция сеянцев моделью YOLO (`.pt`);
- классификация частей внутри найденных объектов;
- поворот изображений и кропов с пересчетом `bbox`;
- дерево слоев, статистика и миниатюры;
- сохранение PDF-отчета.

## Что убрано

- ONNX и `onnxruntime`;
- экспорт в `JSON/CSV/COCO/YOLO/annotated`;
- кэш, история измерений и калибровка;
- настройки, локализация и реестр моделей;
- `qt-material` и файлы сборки.

## Установка

```bash
python -m pip install -e .
```

Или:

```bash
python -m pip install -r seeding/requirements.txt
python -m pip install -r requirements-dev.txt
```

## Запуск

```bash
python -m seeding.main
```

С явными путями к моделям:

```bash
python -m seeding.main --weights models/bestCrop.pt --classify-weights models/bestKlassSeg.pt
```

## Переменные окружения

- `YOLO_WEIGHTS_PATH` - путь к модели детекции.
- `YOLO_CLASSIFY_WEIGHTS_PATH` - путь к модели классификации.

## Горячие клавиши

- `Ctrl+O` - открыть файлы.
- `Ctrl+Shift+O` - добавить файлы.
- `Ctrl+F` - детекция на текущем изображении.
- `Ctrl+Shift+F` - детекция на всех страницах.
- `Ctrl+C` - классификация частей.
- `Ctrl+R` - поворот.
- `Ctrl+P` - PDF-отчет.
- `Ctrl++`, `Ctrl+-`, `Ctrl+0` - масштаб.

## Структура

```text
seeding/
  main.py
  config.py
  controllers.py
  inference.py
  models.py
  report.py
  services.py
  requirements.txt
  utils/
  ui/
  resources/
tests/
```

## Тесты

```bash
python -m pytest tests
```
