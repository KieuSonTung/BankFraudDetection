import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import roc_curve
import warnings
from src.preprocess import utils
from src.visualize import plot_feat_imp

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

class XGBoostModel:
    def optimize(self, X_train, y_train, n_trials=5):
        def objective(trial, data, target):
            train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)

            param = {
                'random_state': 48,
                'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                'learning_rate': trial.suggest_float('learning_rate', 0.006, 0.05),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_child_weight': trial.suggest_int('min_child_samples', 1, 30)
            }


            model = XGBClassifier(**param, verbosity=0)  
            model.fit(
                train_x, train_y,
                eval_set=[(test_x,test_y)],
                early_stopping_rounds=100,
                verbose=False
            )
            
            preds = model.predict_proba(test_x)[:, 1]
            
            fprs, tprs, thresholds = roc_curve(test_y, preds)
            tpr = tprs[fprs < 0.05][-1]
            
            return tpr
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

        print('Number of finished trials:', len(study.trials))
        
        return study.best_trial.params
    
    def retrain_kfold(self, X_train, y_train, X_test, y_test, best_params, n_splits=5):        
        model_fi = 0
        total_mean_tpr = 0
        y_pred_ls, y_prob_ls = [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = XGBClassifier(**best_params, verbosity=0, n_jobs=-1)
        
        for num, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
            train_x, val_x = X_train.loc[train_idx], X_train.loc[valid_idx]
            train_y, val_y = y_train.loc[train_idx], y_train.loc[valid_idx]
            
            model.fit(
                train_x, train_y,
                eval_set=[(train_x, train_y), (val_x, val_y)],
                eval_metric='auc', 
                early_stopping_rounds=100,
                verbose = False
            )
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred_ls.append(y_pred)
            y_prob_ls.append(y_prob)
            
            fold_tpr = utils.custom_tpr(y_test, y_prob)
            
            model_fi += model.feature_importances_ / n_splits
            print(f'Fold {num} recall: {fold_tpr}')
            
            total_mean_tpr += fold_tpr / n_splits

        print(f'Mean TPR on infer set: {total_mean_tpr}')

        # plot feature importace
        plot_feat_imp(X_train, model_fi)

        return y_pred_ls, y_prob_ls


    def fit(self, df, month_pred):
        '''
        df: dataframe, contains both train and test set
        voting_method: voting method (options: soft, hard; default: soft)
        '''
        
        p = utils.PreProcess()
        X_train, y_train, X_test, y_test = p.fit(df, month_pred)

        best_params = self.optimize(X_train, y_train)

        y_pred_ls, y_prob_ls = self.retrain_kfold(X_train, y_train, X_test, y_test, best_params)

        prob = utils.soft_voting(y_prob_ls)
        pred = utils.hard_voting(y_pred_ls)

        return pred, prob

        