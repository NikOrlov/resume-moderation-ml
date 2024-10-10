from datetime import datetime

from ml_tools.kardinal_tools.utils.config import Config


class ModerationConfig(Config):
    __slots__ = ('hh_db_config', 'creation_time_threshold', 'test_size', 'vectorizer_size',
                 'model_params', 'drop_salary_params', 'bad_positions', 'cv_seed',
                 'cv_number_of_folds', 'common_xgb_params', 'hp_param_space', 'hp_max_evals')

    def __init__(self):
        super(ModerationConfig, self).__init__()
        self.hh_db_config = {
            'database': 'hh',
            'host': 'dev-db2.pyn.ru',
            'port': 5432,
            'user': 'hh',
            'password': '123'
        }
        self.creation_time_threshold = '2019-01-01'
        self.test_size = 0.3
        # 20 % of resumes are only used to fit vectorizer. These resumes never used for fitting the model.
        self.vectorizer_size = 0.2
        self.model_params = {
            'approve_complete': {
                'colsample_bytree': 0.6,
                'max_delta_step': 2.0,
                'min_child_weight': 0.55,
                'num_boost_round': 1000,
                'subsample': 0.9,
                'eta': 0.09,
                'max_depth': 12,
                'gamma': 1.95,
                'verbosity': 1,
                'threshold': {
                    'precision': 0.90
                }
            },
            'approve_incomplete': {
                'colsample_bytree': 0.8,
                'max_delta_step': 0.0,
                'min_child_weight': 1.1,
                'num_boost_round': 1000,
                'subsample': 0.95,
                'eta': 0.09,
                'max_depth': 12,
                'gamma': 1.45,
                'verbosity': 1,
                'threshold': {
                    'precision': 0.97
                }
            },
            'block': {
                'colsample_bytree': 0.9,
                'max_delta_step': 0.0,
                'min_child_weight': 1.05,
                'num_boost_round': 1000,
                'subsample': 0.75,
                'eta': 0.018,
                'max_depth': 14,
                'gamma': 1.95,
                'verbosity': 1,
                'threshold': {
                    'recall': 0.20
                }
            },
            'careless_key_skill_information': {
                'colsample_bytree': 0.95,
                'max_delta_step': 0.0,
                'min_child_weight': 1.35,
                'num_boost_round': 260,
                'subsample': 0.9,
                'eta': 0.031,
                'max_depth': 5,
                'gamma': 0.75,
                'verbosity': 1,
                'threshold': {
                    'recall': 0.95
                }
            },
            'careless_additional_information': {
                'colsample_bytree': 0.85,
                'max_delta_step': 1.5,
                'min_child_weight': 1.35,
                'num_boost_round': 1000,
                'subsample': 1.0,
                'eta': 0.015,
                'max_depth': 15,
                'gamma': 1.8,
                'verbosity': 1,
                'threshold': {
                    'precision': 0.9
                }
            },
            'bad_function': {
                'colsample_bytree': 0.95,
                'max_delta_step': 0.0,
                'min_child_weight': 1.1,
                'num_boost_round': 990,
                'subsample': 0.9,
                'eta': 0.014,
                'max_depth': 7,
                'gamma': 1.2,
                'verbosity': 1,
                'threshold': {
                    'precision': 0.75
                }
            },
            'bad_education': {
                'colsample_bytree': 0.65,
                'max_delta_step': 0.0,
                'min_child_weight': 1.15,
                'num_boost_round': 990,
                'subsample': 0.9,
                'eta': 0.01,
                'max_depth': 11,
                'gamma': 0.55,
                'verbosity': 1,
                'threshold': {
                    'precision': 0.75
                }
            }
        }

        self.drop_salary_params = {
            'xgb_params': {
                'num_boost_round': 500,
                'eta': 0.2,
                'gamma': 0.0,
                'max_delta_step': 1,
                'max_depth': 3,
                'min_child_weight': 30,
                'subsample': 0.9,
                'verbosity': 1,
            },
            'threshold': {
                'precision': 0.97
            },
            'validation_size': 0.3
        }

        self.bad_positions = {
            'seed': 11226,
            'subset_size': 1.0,
            'log_reg': {
                'C': 11.24596,
            }
        }
        self.cv_seed = 18250
        self.cv_number_of_folds = 3
        self.common_xgb_params = {
            'seed': 60011,
        }

        try:
            from hyperopt import hp
            self.hp_param_space = {
                'num_boost_round': 100,
                'eta': hp.quniform('eta', 0.01, 0.5, 0.01),
                'max_depth': hp.choice('max_depth', range(1, 15)),
                'min_child_weight': hp.quniform('min_child_weight', 0.5, 1.5, 0.05),
                'max_delta_step': hp.qloguniform('max_delta_step', -6, 2.19722, 0.5),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'gamma': hp.quniform('gamma', 0.0, 2.0, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                'seed': 60011  # for reproducibility of experiments
            }
        except ImportError:
            self.hp_param_space = {}

        self.hp_max_evals = 50

    @staticmethod
    def moderated_by_human(resume_id, creation_time):
        # 2021-06-18 процент валидационной выборки был уменьшен до 5 - смотри HH-131360
        border5 = datetime.strptime('2021-06-18 00:00:00', '%Y-%m-%d %H:%M:%S')
        percent = 10 if creation_time < border5 else 5
        return (resume_id % 100) < percent
