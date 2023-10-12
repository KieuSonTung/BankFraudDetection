import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import roc_curve
from src.preprocess import utils
from src.visualize.plot_feat_imp import plot_feature_importance


class LGBMModel:
    def optimize(self, X_train, y_train, n_trials=1):
        def objective(trial, data, target):
            train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)

            param = {
                'random_state': 48,
                'early_stopping_round': 200,
                'verbose': -1,
                'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 5000, 10000),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
                'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                'learning_rate': trial.suggest_float('learning_rate', 0.006, 0.02),
                'max_depth': trial.suggest_int('max_depth', 10, 100),
                'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
            }

            model = LGBMClassifier(**param)  
            model.fit(
                train_x, train_y,
                eval_set=[(test_x,test_y)]
            )
            
            preds = model.predict_proba(test_x)[:, 1]
            
            tpr = utils.custom_tpr(test_y, preds)
            
            return tpr
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

        print('Number of finished trials:', len(study.trials))

        best_params = study.best_trial.params
        print('Best params', best_params)
        
        return best_params
    
    def retrain_kfold(self, X_train, y_train, X_test, y_test, best_params, n_splits=2):        
        model_fi = 0
        total_mean_tpr = 0
        y_pred_ls, y_prob_ls = [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = LGBMClassifier(**best_params, verbose=-1, n_jobs=-1)
        
        for num, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
            train_x, val_x = X_train.loc[train_idx], X_train.loc[valid_idx]
            train_y, val_y = y_train.loc[train_idx], y_train.loc[valid_idx]
            
            model.fit(
                train_x, train_y,
                eval_set=[(train_x, train_y), (val_x, val_y)],
                eval_metric='auc'
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
        plot_feature_importance(X_train, model_fi)

        return y_pred_ls, y_prob_ls


    def fit(self, df, month_pred):
        '''
        df: dataframe, contains both train and test set
        voting_method: voting method (options: soft, hard; default: soft)
        '''
        
        p = utils.PreProcess()
        X_train, y_train, X_test, y_test = p.fit(df, month_pred)

        best_params = self.optimize(X_train, y_train)
        # best_params = {'n_estimators': 8902, 'reg_alpha': 4.225210678586751, 'reg_lambda': 0.4564306389521886, 'colsample_bytree': 0.5, 'subsample': 0.7, 'learning_rate': 0.008065204720928393, 'max_depth': 76, 'num_leaves': 824, 'min_child_samples': 101, 'min_data_per_groups': 63}

        y_pred_ls, y_prob_ls = self.retrain_kfold(X_train, y_train, X_test, y_test, best_params)

        prob = utils.soft_voting(y_prob_ls)
        pred = utils.hard_voting(y_pred_ls)

        result = X_test.copy()
        result['y_true'] = y_test
        result['y_pred'] = pred
        result['y_prob'] = prob
        result['month'] = month_pred

        return result

        


