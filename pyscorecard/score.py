# -*- coding: utf8 -*-
import math
import pandas as pd
from .Tool import Bins


def logit_trans(x):
    p = 1 / (math.e ** (-x) + 1)
    return p


def scoring(df: pd.DataFrame, y, change_report: pd.DataFrame, var_score: dict, const):
    """
    card列: Feature	Interval	woe	coef	score
    :param df:
    :param y:
    :param change_report:
    :param var_score:
    :param const:
    :return:
    """
    bins = Bins()
    all_report_new = bins.mannual_rebin(df, change_report, df[y])
    result = bins.whole_woe_replace(df, all_report_new)

    def calc_model_score(x):
        s = 0
        for v in var_score.keys():
            s += x[v] * var_score[v]
        s += const
        p = logit_trans(s)
        return p

    y_pred = result.apply(calc_model_score, axis=1)
    return y_pred
