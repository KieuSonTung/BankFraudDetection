import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from src.preprocess import utils, evaluate
from src.visualize.plot_feat_imp import plot_feature_importance


class RandomForestModel:

    def retrain_kfold(self, X_train, y_train, X_test, y_test, n_splits=2):        
        model_fi = 0
        total_mean_tpr = 0
        y_pred_ls, y_prob_ls = [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = RandomForestClassifier(verbose=-1, n_jobs=-1)
        
        for num, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
            train_x, val_x = X_train.loc[train_idx], X_train.loc[valid_idx]
            train_y, val_y = y_train.loc[train_idx], y_train.loc[valid_idx]
            
            model.fit(
                train_x, train_y
            )
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred_ls.append(y_pred)
            y_prob_ls.append(y_prob)
            
            fold_tpr = evaluate.custom_tpr(y_test, y_prob)
            
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

        
        y_pred_ls, y_prob_ls = self.retrain_kfold(X_train, y_train, X_test, y_test)

        prob = utils.soft_voting(y_prob_ls)
        pred = utils.hard_voting(y_pred_ls)

        # print classification report
        evaluate.test_classifier(y_test, pred, prob)

        result = pd.DataFrame(X_test['id'])
        result['y_true'] = y_test
        result['y_pred'] = pred
        result['y_prob'] = prob
        result['month'] = month_pred

        return result

        