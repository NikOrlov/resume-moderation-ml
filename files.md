- (+) targets.py
    определение класса таргета - ModerationTargets
- (+) source.py
  считываем выборку, разбиваем на две части (кэшируем результат - cache/train)
  - main - для основной модели (main_resumes - список словарей, main_targets - ModerationTargets)
  - vectorizer - для модели bad-position (vectorizer_resumes, vectorizer_targets)

- (+) bad_positions.py (используется для vectorizer)
  отдельная модель на тайтлайх (логрег) 
- Данные берем из resume_from_hive_vectorizer.csv
  сохранили в кэш (cache/bad_positions) 
  data - {titles: list, targets: ndarray}
  model - обученный пайплайн
    Пример использования:
    - model.predict_proba([{'title': 'mechanical technician'}])
      array([[0.34752928, 0.65247072]])

    - model.predict_proba([{'title': 'городской планировщик'}])
    array([[0.97294971, 0.02705029]])

- (+) config.py
- (+) data
  изначально храним данные для обучения
  - resume_deleted_from_db.csv
  - resume_from_hive.csv

- (+ / -) dropsalary.py (отрефакторить @classifier.store())
  отдельная модель (XGBClassifier) - cache/dropsalary/*
  
  data - {resumes: list, targets: ndarray, weights: ndarray} (пример резюме {'salary': {'amount': 2000, 'currency': 'BYR'}})
  vectorizer - обученный пайплайн 
  resume_vectors - nd.array (построчно [2.00000000e+03 5.87337014e+04 4.00000000e+00 3.00000000e+00, 7.50000000e-01 1.00000000e+00 2.50000000e-01])
  classifier - {titles: list, targets: ndarray}



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

## Модели:
- approve_complete
- approve_incomplete
- block
- careless_key_skill_information
- careless_additional_information
- bad_function
- bad_education
