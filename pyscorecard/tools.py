# -*- coding: utf8 -*-
import pandas as pd
import pickle
import math
import pyecharts.options as opts
from copy import copy
from functools import reduce
from pyecharts.charts import Line, HeatMap
from pyecharts.globals import ThemeType
from pyecharts.globals import CurrentConfig, NotebookType
from sklearn.utils import shuffle


def filter_by_treshhold(combine_result: pd.DataFrame, condition: dict, how: str) -> list:
    """
    按照条件过滤给定的dataframe
    combine_result = self.get_combine_result(xx,xx,xx)
    :param combine_result: dataframe
    :param condition:
        condition中op对应的操作有:
        (需要严格按照下面的格式来! *_*!!)
            eq: =
            ne: !=
            le: <=
            lt: <
            ge: >=
            gt: >
        condition =  {
            "var1": {"op": "ge", "v": xxxx},
            "var2": {"op": "le", "v": xxxx},
            "var3": {"op": "eq", "v": xxxx},
            "var4": {"op": "gt", "v": xxxx},
            "var5": {"op": "lt", "v": xxxx},
            "var6": {"op": "ge", "v": xxxx},
        }
    :param: how: 这个参数和pd.merge的how相同, 交集: inner, 并集:outer
    :return:
    """
    dfs = []
    for var in condition.keys():
        op = condition[var]["op"]
        v = condition[var]["v"]
        filter_list = getattr(combine_result[var], op)(v)
        df = combine_result[filter_list]
        dfs.append(df)
    final = reduce(lambda left, right: pd.merge(left, right, on='var_code', how=how), dfs)
    return final


def save_to_pkl(your_obj: object, filename: str):
    """
    对象序列化
    :param your_obj:
    :param filename:
    :return:
    """
    with open(filename, "wb") as f:
        pickle.dump(your_obj, f)


def load_from_pkl(filename: str):
    """
    反序列化
    :param filename:
    :return:
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def dict2file(data: dict, filepath: str):
    """
    dict to file
    :param data:
    :param filepath:
    :return:
    """
    print(filepath)
    pd.Series(data).to_json(filepath)


def simple_line(x: list, y: list, y_name: str, title: str):
    """
    pyecharts简单line图
    :param x:
    :param y:
    :param y_name
    :param title:
    :return:
    """
    CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
    CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))\
        .add_xaxis(x)\
        .add_yaxis(y_name, y)\
        .set_global_opts(title_opts=opts.TitleOpts(title=title))
    line.load_javascript()
    return line


def df2csv(df: pd.DataFrame, path, filename, index=False):
    file_path = f"{path}/{filename}"
    df.to_csv(file_path, index=index, encoding="utf8")


def interval_round(itv: pd.Interval, n=3):
    """
    对interval的区间进行round
    :param itv:
    :param n:
    :return:
    """
    if isinstance(itv, pd.Interval):
        return pd.Interval(
            left=round(itv.left, n),
            right=round(itv.right, n),
            closed=itv.closed)
    else:
        return itv


def calc_woe(info: pd.Series):
    """
    计算WOE
    :param info:
    :return:
    """
    try:
        w = math.log(info["1_prop"] / info["0_prop"])
        w = round(w, 4)
    except Exception as e:
        w = None
    return w


def calc_iv(info: pd.Series):
    """
    计算IVC
    :param info:
    :return:
    """
    try:
        a = info["1_prop"]
        b = info["0_prop"]
        w = (a - b) * math.log(a / b)
        w = round(w, 4)
    except Exception as e:
        w = None
    return w


def split_data(df: pd.DataFrame, r=0.7):
    """
    分割数据
    :param df:
    :param r:
    :return:
    """
    df_chaos = shuffle(df)
    split_index = int(len(df_chaos) * r)
    df1 = df_chaos.iloc[0: split_index]
    df2 = df_chaos.iloc[split_index:]
    return df1, df2


def heat_map_common(df: pd.DataFrame, title):
    """
    一般的热力图
    :param df:
    :param title:
    :return:
    """
    df = df.T
    myvalues = []
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            tmp = [i, j, df.iloc[i, j]]
            myvalues.append(copy(tmp))
    heat_map = HeatMap(init_opts=opts.InitOpts(width=1200)) \
        .add_xaxis(list(df.columns)) \
        .add_yaxis("corr", list(df.index), myvalues) \
        .set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        datazoom_opts=[opts.DataZoomOpts(is_show=True, is_realtime=True, range_start=0, range_end=100), ],
        visualmap_opts=opts.VisualMapOpts(pos_right=20, max_=max(df.max())),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(is_scale=True, is_inverse=False,
                                 axislabel_opts=opts.LabelOpts(is_show=True, rotate=-60)),
        yaxis_opts=opts.AxisOpts(is_scale=True, is_inverse=True, position="top",
                                 axislabel_opts=opts.LabelOpts(is_show=True, position="right")),
        tooltip_opts=opts.TooltipOpts(is_show=True)) \
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="insideBottom"))
    return heat_map


def auto_schema(df: pd.DataFrame):
    """
    自动类型推导
         |      b  boolean
         |      i  signed integer
         |      u  unsigned integer
         |      f  floating-point
         |      c  complex floating-point
         |      m  timedelta
         |      M  datetime
         |      O  object
         |      S  (byte-)string
         |      U  Unicode
         |      V  void
    :param df:
    :return:
    """
    cols = df.columns
    for c in cols:
        if df[c].dtype.kind in ["O", "S", "U"]:
            if df[c]._is_numeric_mixed_type:
                df[c] = df[c].astype(float)
        else:
            continue
    return df
