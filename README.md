# t2i-f1-challenge

## 1. Описание решения

Kaggle Text-to-Image Challenge - задача оптимизации object-level F1. Целевой контур:

`Prompt -> POS/Noun Extraction -> Image Generation -> YOLO Detection -> F1`

Ключевая идея: управлять генерацией так, чтобы YOLO детектировал ожидаемые объекты
из текста максимально полно (recall) и без лишних срабатываний (precision).

### Архитектура по модулям

- `PromptAgent`: загружает промпты, извлекает ожидаемые объекты и нормализует их в единый словарь.
- `PromptOptimizationAgent`: переписывает промпт под object grounding (явные объекты, количество, снижение двусмысленности).
- `GenerationAgent`: выполняет детерминированную генерацию изображений с фиксированным seed и обязательным соответствием `prompt_id -> filename`.
- `MultiSamplingAgent`: генерирует несколько кандидатов (seed/guidance/sampler) и выбирает лучший по офлайн-метрике.
- `DetectionAgent`: запускает YOLO-детекцию, применяет пороги и нормализует/дедуплицирует классы.
- `EvaluationAgent`: считает object-level `precision`, `recall`, `F1` для каждого промпта и в среднем по набору.
- `OptimizationAgent`: подбирает конфигурации генерации, realism bias и object-count reinforcement для максимизации F1.
- `FinalizationAgent`: фиксирует лучшую конфигурацию и пересоздает финальный набор (ровно одно изображение на prompt).
- `ExportAgent`: собирает DreamLayer bundle (`.png`, `results.csv`, `config-dreamlayer.json`) и проверяет целостность перед отправкой.

### Правила воспроизводимости

- Все гиперпараметры централизованы в `configs/generation_config.json`.
- Для всех прогонов обязателен фиксированный seed/seed_strategy.
- Имена изображений строго совпадают с `prompt_id` (`0001.png`, `0002.png`, ...).
- `results.csv` не редактируется вручную.
- Экспорт DreamLayer формируется только скриптами проекта.

## 2. Как поднять и воспроизвести

### Шаг 1. Установка окружения

```bash
cd t2i-f1-challenge
python3 -m pip install --upgrade pip
python3 -m pip install uv
uv tool install poetry
uv sync --all-groups
```

Требования: Python `>=3.10,<3.14`, интернет для первой установки, опционально GPU.

### Шаг 2. Подготовка входных данных

- Поместите файл промптов в `prompts/` (например, `prompts/DreamLayer-Prompt-Kaggle.txt`).
- Проверьте `configs/generation_config.json`.
- Убедитесь, что YOLO-веса доступны по пути `models/yolo/yolov8n.pt`
  (или включите загрузку весов в конфиге, если это нужно).

### Шаг 3. Проверка среды

```bash
uv run ./scripts/validate_environment.sh
uv run python scripts/gpu_compatibility_test.py
uv run python scripts/yolo_inference_test.py --weights models/yolo/yolov8n.pt
```

### Шаг 4. Воспроизведение полного пайплайна

```bash
# 1) Базовая генерация
.venv/bin/python generation/generate_baseline.py --config configs/generation_config.json

# 2) Поиск лучших гиперпараметров
.venv/bin/python evaluation/hyperparameter_sweep.py --config configs/generation_config.json

# 3) Тюнинг реализма и count-reinforcement
.venv/bin/python evaluation/realism_bias_tuning.py --config configs/generation_config.json
.venv/bin/python evaluation/object_count_reinforcement.py --config configs/generation_config.json

# 4) Финальная генерация и сборка DreamLayer bundle
.venv/bin/python generation/generate_final.py --config configs/generation_config.json
.venv/bin/python generation/generate_report_bundle.py --config configs/generation_config.json

# 5) Валидация готового bundle
.venv/bin/python generation/generate_report_bundle.py --config configs/generation_config.json --validate-only
```

### Шаг 5. Проверка перед push

```bash
.venv/bin/pytest -q tests
git status
```

Подробная инструкция по запуску/тестам: `INSTALL_RUN_TEST.md`.
