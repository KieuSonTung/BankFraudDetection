from src.model.basemodel import BaseModel
from xgboost import XGBClassifier
from optuna.trial import Trial


class XGBoostModel(BaseModel):
    def get_classifier(self, param):
        return XGBClassifier(**param, verbosity=0, n_jobs=-1)

    def get_model_params(self, trial: Trial):
        return {
            'random_state': 48,
            'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.006, 0.05),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_child_weight': trial.suggest_int('min_child_samples', 1, 30)
        }