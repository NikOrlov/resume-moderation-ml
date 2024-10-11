- (+) targets.py
    определение класса  таргета - ModerationTargets
- (+) source.py
  считываем выборку, разбиваем на две части (кэшируем результат - cache/resume_moderation_ml/model/train)
    main - для основной модели (raw_resumes - список словарей, targets - ModerationTargets)
    vectorizer - для модели bad-position (vectorizer_resumes, vectorizer_targets)

- (+) bad_positions.py (используется для vectorizer)
  отдельная модель на тайтлайх (логрег) 
  сохранили в кэш (cache/resume_moderation_ml/model/train/bad_positions) 
  data - {titles: list, targets: ndarray}
  model - обученный пайплайн

- (+) config.py
- (+) data
  изначально храним данные для обучения
  - resume_deleted_from_db.csv
  - resume_from_hive.csv

- (+ / -) dropsalary.py (отрефакторить @classifier.store())
  отдельная модель (XGBClassifier) - resume_moderation_ml/model/classifier/dropsalary/vectorizer.pickle
  data - {resumes: list, targets: ndarray, weights: ndarray} - cache/resume_moderation_ml/model/train/dropsalary/data.pickle
  X = взяли сырые резюме + прогнали через векторайзер (taget == 1, у кого ЗП низкая)
  y, w
    сохранили в кэш (resume_moderation_ml/model/classifier/dropsalary) 
    classifier - {titles: list, targets: ndarray}
    vectorizer - обученный пайплайн

- (+) vocabulary.py
    создаем словари - маппинги текста в инт:
    education
    experience
    name
    recommendation
    skill
    title
    university

- (+) environment.py - удалил

- (+) vectorize.py
    создаем фичи резюме (cache/resume_moderation_ml/model/train/resume_vectors.pickle) csr_matrix(resumes, 5286)
    создаем векторайзер (resume_moderation_ml/model/classifier/vectorizer.pickle)
- 
- (+) model.py
    подготовка данных для обучения каждой задачи (get_task_subjects(task_name, resume_vectors, targets))
    создание модели (create_model(task_name, manual_feature_number=0, xgb_parameters=None))

- (+ / -) fit.py - нужна большая выборка для тренировки (подсчет roc_auc_score) для `careless_additional_information`
    обучаем модели под каждую таску и сохраняем в кэше (resume_moderation_ml/model/classifier/*.pickle)

- (?) evaluate.py - зачем нужен?

- (?) optimize.py - зачем нужен?
    Возможно такой пайплайн:

    - Нашли лучшие параметры для бустинга
    - Сохранили полученные параметры в конфиг
    - Узнали по фиксированным параметрам скоры
    - Обучили модель для прода по оптимальным параметрам

- utils

- logger.py
- xgb.py
