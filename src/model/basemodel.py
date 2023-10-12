from abc import ABC
import optuna
from optuna.trial import Trial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve
from src.preprocess import utils


class BaseModel(ABC):

    def get_classifier(self, param):
        raise NotImplementedError("Subclasses must implement this method")

    def get_model_params(self, trial: Trial):
        raise NotImplementedError("Subclasses must implement this method")

    def optimize(self, X_train, y_train, n_trials=5):
        def objective(trial: Trial, data, target):
            train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)
            param = self.get_model_params(trial)
            model = self.get_classifier(param)  
            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)
            preds = model.predict_proba(test_x)[:, 1]
            
            tpr = utils.custom_tpr(test_y, preds)
            return tpr

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
        self.best_params = study.best_trial.params
    
    def retrain_kfold(self, X_train, y_train, X_test, y_test, best_params, n_splits=5):        
        model_fi = 0
        total_mean_tpr = 0
        y_pred_ls, y_prob_ls = [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = self.get_classifier(self.best_params)
        
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
            
            fold_tpr = self.custom_tpr(y_test, y_prob)
            
            model_fi += model.feature_importances_ / n_splits
            print(f'Fold {num} recall: {fold_tpr}')
            
            total_mean_tpr += fold_tpr / n_splits

        print(f'Mean TPR on infer set: {total_mean_tpr}')

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