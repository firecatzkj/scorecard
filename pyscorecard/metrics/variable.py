# -*- coding: utf8 -*-
import pandas as pd
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Bar, Line, Grid, Page
from pyecharts.components import Table
from ..binning import BinningInfo


def table_base(df: pd.DataFrame):
    table = Table()

    # headers = ["City name", "Area", "Population", "Annual Rainfall"]
    # rows = [
    #     ["Brisbane", 5905, 1857594, 1146.4],
    #     ["Adelaide", 1295, 1158259, 600.5],
    #     ["Darwin", 112, 120900, 1714.7],
    #     ["Hobart", 1357, 205556, 619.5],
    #     ["Sydney", 2058, 4336374, 1214.8],
    #     ["Melbourne", 1566, 3806092, 646.9],
    #     ["Perth", 5386, 1554769, 869.4],
    # ]
    headers = list(df.columns)
    rows = [list(df.iloc[i]) for i in range(len(df))]
    table.add(headers, rows)
    return table


def bin_plot(bininfo: pd.DataFrame, varname, varname_ch):
    bininfo = bininfo.iloc[: -1].copy()
    bininfo["badRate"] = bininfo["badRate"].apply(lambda x: round(x, 2))
    bininfo["goodDistr"] = bininfo["goodDistr"].apply(lambda x: round(x, 2))
    bininfo["badDistr"] = bininfo["badDistr"].apply(lambda x: round(x, 2))

    bar = Line() \
        .add_xaxis(list(bininfo["cutpoints"])) \
        .add_yaxis("badRate", list(bininfo["badRate"])) \
        .add_yaxis("goodDistr", list(bininfo["goodDistr"]), stack="X") \
        .add_yaxis("badDistr", list(bininfo["badDistr"]), stack="X") \
        .set_global_opts(yaxis_opts=opts.AxisOpts(name="Rate", name_gap=30, name_location="center")) \
        .set_global_opts(xaxis_opts=opts.AxisOpts(name="CutPoint", name_gap=30, name_location="end")) \
        .set_global_opts(title_opts=opts.TitleOpts(title=varname, subtitle=varname_ch)) \
        .set_global_opts(legend_opts=opts.LegendOpts(pos_right="100px", pos_top="20px"))
    return bar


def trans_func(bininfo: pd.DataFrame):
    bininfo = bininfo.iloc[: -1].copy()
    bininfo["badRate"] = bininfo["badRate"].apply(lambda x: round(x, 4))
    bininfo["goodDistr"] = bininfo["goodDistr"].apply(lambda x: round(x, 4))
    bininfo["badDistr"] = bininfo["badDistr"].apply(lambda x: round(x, 4))
    return bininfo


def bin_compare_plot(bininfo_train: pd.DataFrame, bininfo_test: pd.DataFrame, varname, varname_ch):
    bininfo_train = trans_func(bininfo_train)
    bininfo_test = trans_func(bininfo_test)

    bar = Line() \
        .add_xaxis(list(bininfo_train["cutpoints"])) \
        .add_yaxis("Train(训练集)", list(bininfo_train["badRate"])) \
        .add_yaxis("Test(测试集)", list(bininfo_test["badRate"])) \
        .set_global_opts(yaxis_opts=opts.AxisOpts(name="badRate(坏账率)",
                                                  name_gap=40,
                                                  name_location="center")) \
        .set_global_opts(xaxis_opts=opts.AxisOpts(name="CutPoint(分割点)",
                                                  name_gap=30,
                                                  name_location="center",
                                                  axislabel_opts=opts.LabelOpts(rotate=-30))) \
        .set_global_opts(
            title_opts=opts.TitleOpts(title=varname,
                                      subtitle=varname_ch),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            legend_opts=opts.LegendOpts(pos_left="100px", pos_top="20px")
        )
    # bar.render()
    return bar


def var_bin_diff_target(df: pd.DataFrame, x: str, desc_ch: str, cuts: list, y1: str, y2: str, filepath:str=None):
    """
    看变量在y1, y2上的bin的情况对比
    :param df: df
    :param x: 变量
    :param desc_ch: 变量中文
    :param cuts: 切分点
    :param y1:
    :param y2:
    :return:
    """
    bif1 = BinningInfo(df, x, y1, cuts)
    bif2 = BinningInfo(df, x, y2, cuts)
    bar1 = bin_plot(bif1.bininfo(), desc_ch)
    bar2 = bin_plot(bif2.bininfo(), desc_ch)
    grid = Grid(init_opts=opts.InitOpts(width="1200px"))\
        .add(bar1, grid_opts=opts.GridOpts(pos_left="60%", pos_top="100px"), grid_index=0)\
        .add(bar2, grid_opts=opts.GridOpts(pos_right="60%", pos_top="100px"), grid_index=0)
    if filepath:
        grid.render(filepath)
    return grid


# 计算PSI
def psi_var(score_train, score_test, section_num=10):
    """
    @author: ZhaiFeifei
    :param score_train:
    :param score_test:
    :param section_num:
    :return:
    """
    score_train = pd.DataFrame({"0": score_train})
    score_test = pd.DataFrame({"0": score_test})

    total_train_num = len(score_train)
    total_test_num = len(score_test)

    sorted_score_train = score_train.sort_values(by="0")

    PSI_value = 0
    for i in range(0, section_num):
        lower_bound = sorted_score_train.iloc[int(round(total_train_num * (i) / section_num))]
        higher_bound = sorted_score_train.iloc[int(round(total_train_num * (i + 1) / section_num)) - 1]
        score_train_percent = len(
            np.where((score_train >= lower_bound) & (score_train <= higher_bound))[0]) / total_train_num
        score_test_percent = len(
            np.where((score_test >= lower_bound) & (score_test <= higher_bound))[0]) / total_test_num
        PSI_value += (score_test_percent - score_train_percent) * np.log(score_test_percent / score_train_percent)
    return PSI_value


def calc_bin_prop(sub: pd.DataFrame, cuts, dtype):
    """
    计算给定df的每个bin的占比
    :param subdf: pd.DataFrame.groupby
    :param cuts:
    :return:
    """
    is_numeric = str(dtype).startswith("int") or str(dtype).startswith("float") or (str(dtype) == "C")
    if is_numeric:
        cut_category = pd.cut()
    else:
        pass







# 变量趋势: bin_badrate~时间
# 变量趋势: IV~时间