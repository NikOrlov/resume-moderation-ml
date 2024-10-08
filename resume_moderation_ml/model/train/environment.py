# from hhkardinal import db
from resume_moderation_ml.model.train.config import ModerationConfig
# from hhkardinal.train.environment import init_train_env_base
from ml_tools.kardinal_tools.state import State
from ml_tools.kardinal_tools.utils.functions import run_once


@run_once
def init_train_env():
    config = ModerationConfig()
    try:
        from config_override import override
        override(config)
    except ImportError:
        pass
    # init_train_env_base(config)
    # state._initialize(config)
    # db.add_db_config('hh', config.hh_db_config)
