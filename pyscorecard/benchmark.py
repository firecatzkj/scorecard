# -*- coding: utf8 -*-
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


class XGBTrainer:

    def __init__(self, data_x: pd.DataFrame, data_y: pd.Series, grid=False, grid_params=None):
        """
        初始化
        :param data_x:
        :param data_y:
        :param grid:
        """
        self.data_x = data_x
        self.data_y = data_y
        self.grid = grid
        self.grid_params = grid_params
        self.model = None

    def fit(self):
        if self.grid:
            optimal_params = {
                    'learning_rate': 0.1,
                    'n_estimators': 1000,
                    'max_depth': 4,
                    'min_samples_leaf': len(self.data_x) * 0.03}
            xgtrain = xgb.DMatrix(self.data_x.values, label=self.data_y.values)

            cvresult = xgb.cv(
                optimal_params,
                xgtrain,
                num_boost_round=optimal_params['n_estimators'],
                nfold=5,
                metrics='auc',
                early_stopping_rounds=100)
            optimal_params['n_estimators'] = cvresult.shape[0]

            param = {'learning_rate': [0.01, 0.1],
                     'n_estimators': [int(optimal_params['n_estimators']), ],
                     'max_depth': [3, 4, 5],
                     'min_samples_leaf': [int(len(self.data_x) * 0.03),
                                          int(len(self.data_x) * 0.04),
                                          int(len(self.data_x) * 0.05)]
                     }

            cv_obj = GridSearchCV(XGBClassifier(), param, scoring='roc_auc', cv=5)
            cv_obj.fit(self.data_x.astype(float), self.data_y.astype(int))

            final_params = cv_obj.best_params_
            xgb_obj = XGBClassifier(learning_rate=final_params['learning_rate'],
                                    max_depth=final_params['max_depth'],
                                    min_samples_leaf=final_params['min_samples_leaf'],
                                    n_estimators=final_params['n_estimators'])

            xgb_obj.fit(self.data_x, self.data_y)
            self.model = xgb_obj

        else:
            model = XGBClassifier()
            model.fit(self.data_x, self.data_y)
            self.model = model

    def predict(self, data_x: pd.DataFrame):
        cmodel = self.model
        prob_res = cmodel.predict_proba(data_x)
        # 可以调用clf.classes_ 来确定prob对应那一类,因为这里主要是01分类,所以默认返回1的probity
        return [i[1] for i in prob_res]

    def dissect_model(self):
        cmodel = self.model
        return cmodel.get_booster().trees_to_dataframe()
