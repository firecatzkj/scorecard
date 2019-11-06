# -*- coding: utf8 -*-
"""
数据预处理功能函数
"""
import pandas as pd
import math


def numeric_outlier_transform(var: pd.Series, floor: float=0.01, roof: float=0.99):
    """
    原变量值<1%分位点 用1%分位点代替, >99%分位点用99%分位点代替
    :param var:
    :param floor:
    :param roof:
    :return:
    """
    value_floor = pd.Series(var).quantile(floor)
    value_roof = pd.Series(var).quantile(roof)

    def inner_trans(x):
        if x < value_floor:
            return value_floor
        elif x > value_roof:
            return value_roof
        else:
            return x

    var_trans = pd.Series(var).map(inner_trans)
    return var_trans


def category_logit_transform(df: pd.DataFrame, varname: str, y: str):
    """
    类别型变量对每一类做logit转换, category ==> numeric
    todo: 目前还没做异常处理
    :param df:
    :param varname:
    :param y:
    :return:
    """
    this_df = df[[varname, y]]
    total_good = len(df[df[y] == 0])
    total_bad = len(df[df[y] == 1])
    trans_dict = {}
    for sub in this_df.groupby(by=varname):
        sub_df = pd.DataFrame(sub[1])
        good = len(sub_df[sub_df[y] == 0])
        bad = len(sub_df[sub_df[y] == 1])
        good_distr = good / total_good
        bad_distr = bad / total_bad
        logit_value = math.log(bad_distr / good_distr)
        trans_dict[sub[0]] = round(logit_value, 6)

    def inner_trans(x):
        return trans_dict[x]

    var_trans = pd.Series(df[varname]).map(inner_trans)
    return var_trans
