# resume-moderation-ml

## Модели

- `approve_complete`
- `approve_incomplete`
- `block`
- `careless_key_skill_information`
- `careless_additional_information`
- `bad_function`
- `bad_education`


## Файлы и их назначения

### targets.py
- **Определение класса таргета**: `ModerationTargets`

### source.py
- **Считывание выборки и разбиение на две части** (кэширование результата в `cache/resume_moderation_ml/model/train`):
  - `main`: для основной модели (`raw_resumes` - список словарей, `targets` - `ModerationTargets`)
  - `vectorizer`: для модели bad-position (`vectorizer_resumes`, `vectorizer_targets`)

### bad_positions.py
- **Отдельная модель на тайтлах** (логистическая регрессия)
- **Кэширование**: `cache/resume_moderation_ml/model/train/bad_positions`
- **Данные**: `{titles: list, targets: ndarray}`
- **Модель**: обученный пайплайн

### config.py
- Конфигурационные файлы

### data
- **Изначальное хранение данных для обучения**:
  - `resume_deleted_from_db.csv`
  - `resume_from_hive.csv`

### dropsalary.py
- **Отдельная модель** (XGBClassifier) - `resume_moderation_ml/model/classifier/dropsalary/vectorizer.pickle`
- **Данные**: `{resumes: list, targets: ndarray, weights: ndarray}` - `cache/resume_moderation_ml/model/train/dropsalary/data.pickle`
- **Обработка**: сырые резюме + векторайзер (target == 1, у кого ЗП низкая)
- **Кэширование**: `resume_moderation_ml/model/classifier/dropsalary`
- **Классификатор**: `{titles: list, targets: ndarray}`
- **Векторайзер**: обученный пайплайн

### vocabulary.py
- **Создание словарей** - маппинги текста в int:
  - `education`
  - `experience`
  - `name`
  - `recommendation`
  - `skill`
  - `title`
  - `university`

### environment.py
- **Удален**

### vectorize.py
- **Создание фич резюме**: `cache/resume_moderation_ml/model/train/resume_vectors.pickle` `csr_matrix(resumes, 5286)`
- **Создание векторайзера**: `resume_moderation_ml/model/classifier/vectorizer.pickle`

### model.py
- **Подготовка данных для обучения каждой задачи**: `get_task_subjects(task_name, resume_vectors, targets)`
- **Создание модели**: `create_model(task_name, manual_feature_number=0, xgb_parameters=None)`

### fit.py
- **Требуется большая выборка для тренировки** (подсчет `roc_auc_score`) для `careless_additional_information`
- **Обучение моделей под каждую задачу и кэширование**: `resume_moderation_ml/model/classifier/*.pickle`

### evaluate.py
- **Зачем нужен?**

### optimize.py
- **Зачем нужен?**
  - Возможный пайплайн:
    - Поиск лучших параметров для бустинга
    - Сохранение параметров в конфиг
    - Оценка по фиксированным параметрам
    - Обучение модели для прода по оптимальным параметрам

## Утилиты

- **logger.py**
- **xgb.py**

