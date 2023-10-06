import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import roc_curve
from preprocess import utils


class LGBMModel:

    def process_cat_feats(self, X, object_cols):
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # Get one-hot-encoded columns
        ohe_cols = pd.DataFrame(ohe.fit_transform(X[object_cols]))
        ohe_cols.index = X.index

        num_X = X.drop(object_cols, axis=1)
        
        X = pd.concat([num_X, ohe_cols], axis=1)

        X.columns = X.columns.astype(str)

        return X
    
    def split_train_test_set(self, df):
        # Split data into features and target
        X = df.drop(['fraud_bool'], axis=1)
        y = df['fraud_bool']

        # Train test split by 'month', month 0-5 are train, 6-7 are test data as proposed in the paper
        X_train = X[X['month'] < 6]
        X_test = X[X['month'] >= 6]
        y_train = y[X['month'] < 6]
        y_test = y[X['month'] >= 6]

        X_train.drop('month', axis=1, inplace=True)
        X_test.drop('month', axis=1, inplace=True)

        return X_train, y_train, X_test, y_test
    
    def preproces(self, df):
        df = df.drop(['device_fraud_count'], axis=1, errors='ignore')

        X_train, y_train, X_test, y_test = self.split_train_test_set(df)
        
        # list of column-names and whether they contain categorical features
        s = (X_train.dtypes == 'object') 
        object_cols = list(s[s].index)

        X_train = self.process_cat_feats(X_train)
        X_test = self.process_cat_feats(X_test)

        X_train = X_train.reset_index() \
                .drop(columns='index')
        X_test = X_test.reset_index() \
                .drop(columns='index')
        
        y_train = y_train.reset_index() \
                .drop(columns='index')
        y_test = y_test.reset_index() \
                .drop(columns='index')

        return X_train, y_train, X_test, y_test
        
    def optimize(self, X_train, y_train, n_trials=10):
        def objective(trial, data, target):
            train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)

            param = {
                'random_state': 48,
                'n_estimators': trial.suggest_int('n_estimators', 10000, 20000),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
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
        
    def custom_tpr(self, y_true, y_pred):
        fprs, tprs, thresholds = roc_curve(y_true, y_pred)
        tpr = tprs[fprs < 0.05][-1]
        
        return tpr
    
    def retrain_kfold(self, X_train, y_train, X_test, y_test, best_params, n_splits=5):        
        preds = 0
        model_fi = 0
        total_mean_tpr = 0
        y_pred_ls, y_prob_ls = [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = LGBMClassifier(**best_params)
        
        for num, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
            train_x, val_x = X_train.loc[train_idx], X_train.loc[valid_idx]
            train_y, val_y = y_train.loc[train_idx], y_train.loc[valid_idx]
            
            model.fit(
                train_x, train_y,
                eval_set=[(train_x, train_y), (val_x, val_y)],
                eval_metric='auc', 
                early_stopping_rounds=100
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


    def fit(self, df, voting_method='soft'):
        '''
        df: dataframe, contains both train and test set
        voting_method: voting method (options: soft, hard; default: soft)
        '''
        
        X_train, y_train, X_test, y_test = self.preproces(df)

        best_params = self.optimize(X_train, y_train)

        y_pred_ls, y_prob_ls = self.retrain_kfold(X_train, y_train, X_test, y_test, best_params)

        if voting_method == 'soft':
            pred = utils.soft_voting(y_prob_ls)
        elif voting_method == 'hard':
            pred = utils.hard_voting(y_pred_ls)
        else:
            print('TypeError: voting_method must be "soft" or "hard"')

        return pred

        

