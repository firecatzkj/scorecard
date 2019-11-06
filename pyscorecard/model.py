# -*- coding: utf8 -*-
import statsmodels.api as sm
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from copy import copy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, RFE
from .eda import EdaTools


class ModelTools:

    @staticmethod
    def CalVIF(arr, arr_i):
        """
        @author: ZhaiFeifei
        :param arr:
        :param arr_i:
        :return:
        """
        k_vars = arr.shape[1]
        x_i = arr[:, arr_i]
        mask = np.arange(k_vars) != arr_i
        x_noti = arr[:, mask]
        intercept = np.ones(len(x_noti), )
        x_noti = np.column_stack((x_noti, intercept))
        r_squared_i = sm.OLS(x_i, x_noti).fit().rsquared
        vif = 1. / (1. - r_squared_i)
        return vif

    @staticmethod
    def cat_vif(df: pd.DataFrame):
        """
        计算数据集的vif
        :param df: 只有X的数据集
        :return: pd.DataFrame var: vif
        """
        var_all = list(df.columns)
        result = []
        for i in range(len(var_all)):
            tmp = {
                "var": var_all[i],
                "vif": ModelTools.CalVIF(df.values, i)
            }
            result.append(copy(tmp))
        return pd.DataFrame(result)

    @staticmethod
    def ParamEST(results, df_train: pd.DataFrame=None):
        """
        return params estimate from model fit results
        ------------------------------------
        Params
        results: model fit result
        ------------------------------------
        Return
        pandas dataframe
        """

        rlt = pd.concat([
            results.params,
            results.bse,
            results.tvalues,
            (results.params / results.bse) ** 2,
            results.pvalues,
            results.conf_int()
        ], axis=1)
        rlt.columns = [u'参数估计', u'标准误差', u'z值', u'wald卡方', u'p值', u'置信下界', u'置信上界']
        rlt["变量名"] = rlt.index
        rlt = rlt[[
            '变量名', '参数估计', '标准误差', 'z值', 'wald卡方', 'p值', u'置信下界', '置信上界'
        ]]

        if df_train is not None:
            vv = list(rlt["变量名"])
            if "const" in vv:
                vv.remove("const")
            vif_df = ModelTools.cat_vif(df_train[vv])
            rlt = pd.merge(
                rlt,
                vif_df,
                left_on="变量名",
                right_on="var",
                how="left"
            )
            del rlt["var"]
        return rlt

    @staticmethod
    def rfe_select(x, y):
        # 可选择正则项为l2的逻辑回归或者无正则项的逻辑回归（将正则项参数倒数C设置得很大即可实现）
        Lr1 = LogisticRegression(penalty='l2')
        rfe = RFE(Lr1, n_features_to_select=1)
        rfe.fit(x, y)
        rfe_ranking = rfe.ranking_
        return pd.DataFrame({"var": x.columns, "rfe_rank": rfe_ranking})

    @staticmethod
    def rfe_select_cv(x, y):
        # 可选择正则项为l2的逻辑回归或者无正则项的逻辑回归（将正则项参数倒数C设置得很大即可实现）
        Lr1 = LogisticRegression(penalty='l2')
        rfe = RFECV(Lr1, cv=10)
        rfe.fit(x, y)
        rfe_ranking = rfe.ranking_
        return pd.DataFrame({"var": x.columns, "rfecv_rank": rfe_ranking})

    @staticmethod
    def forward_selected(X, y):
        """Linear model designed by forward selection.

        Parameters:
        -----------
        data : pandas DataFrame with all possible predictors and response

        response: string, name of response column in data

        Returns:
        --------
        model: an "optimal" fitted statsmodels linear model
               with an intercept
               selected by forward selection
               evaluated by adjusted R-squared
        """
        response = "y"
        X[response] = y
        data = X
        remaining = set(data.columns)
        remaining.remove(response)
        selected = []
        current_score, best_new_score = 0.0, 0.0
        while remaining:
            print(remaining)
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(response,
                                               ' + '.join(selected + [candidate]))
                score = smf.ols(formula, data).fit().rsquared_adj
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            # if current_score <= best_new_score:
            #     remaining.remove(best_candidate)
            #     selected.append(best_candidate)
            #     current_score = best_new_score
            # if abs(current_score - best_new_score) < 1e-6:
            #     break
            if abs(current_score - best_new_score) < 1e-5:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
            else:
                break
        return selected

    @staticmethod
    def backward_selected(X, y, sls=0.05):
        """
        Linear model designed by backward selection.

        Parameters:
        -----------
        X: pandas DataFrame with all possible predictors
        y: pandas Series with response
        sls: measure for drop variable

        Return:
        --------
        var_list
        """
        # 提取X，y变量名
        var_list = X.columns
        # 首先对所有变量进行模型拟合
        drop_record = []
        while True:
            # mod = sm.Logit(y, sm.add_constant(X[var_list]), missing="drop").fit()
            mod = sm.Logit(y, X[var_list], missing="drop").fit()
            p_list = mod.pvalues.sort_values()
            if p_list[-1] > sls:
                # 提取p_list中最后一个index
                var = p_list.index[-1]
                # var_list中删除
                var_list = var_list.drop(var)
                drop_record.append(var)
            else:
                break
        return list(var_list), drop_record

    @staticmethod
    def check_singular_matrix(x: pd.DataFrame, y: pd.Series):
        """
        检查训练集中存在完全相关的变量
        从而避免model.fit的时候出现LinAlgError: Singular matrix的异常

        import statsmodels.api as sm
        x = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [2, 4, 6],
            "c": [4, 3, 2],
        })
        y = [1, 0, 1]
        sm.Logit(y, x).fit()
        :param x:
        :param y:
        :return:
        """
        exists = False
        try:
            sm.Logit(y, x).fit()
        except np.linalg.LinAlgError as e:
            print("当前训练集中的变量存在完全相关,需要check变量相关性,去除完全相关的变量", e)
            exists = True
        return exists


def mylogit(x: pd.DataFrame, y: pd.Series, add_constant=True, select=None):
    """
    逻辑回归建模
    :param x:
    :param y:
    :param add_constant:
    :param select:
        None: 不选择
        BS: backward
        FS: forward
    :return:
    """
    if select == "BS":
        x_select = ModelTools.backward_selected(x, y)[0]
        x = x[x_select]
    elif select == "FS":
        x_select = ModelTools.forward_selected(x, y)
        x = x[x_select]
    if add_constant:
        x = sm.add_constant(x)
    model = sm.Logit(y, x, missing="drop")
    model_result = model.fit()
    return model_result


def model_predict(model_result, x, add_constant=True):
    """
    模型预测
    :param model_result:
    :param x:
    :param add_constant:
    :return:
    """
    if add_constant:
        x = sm.add_constant(x)
    return model_result.predict(x)


# todo: 做个整体的model describe替代context现有的硬编码
def model_desc(model_result, test, oot):
    raise NotImplementedError
