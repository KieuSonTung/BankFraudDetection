from src.model.basemodel import BaseModel
from lightgbm import LGBMClassifier
from optuna.trial import Trial


class LGBMModel(BaseModel):
    def get_classifier(self, param):
        return LGBMClassifier(**param, verbose=-1, n_jobs=-1)

    def get_model_params(self, trial: Trial):
        return {
            'random_state': 48,
            'early_stopping_round': 200,
            'verbose': -1,
            'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', 10000, 20000),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.006, 0.02),
            'max_depth': trial.suggest_int('max_depth', 10, 100),
            'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
        }