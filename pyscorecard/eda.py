# -*- coding: utf8 -*-
import pandas as pd
import numpy as np
import logging
import warnings
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as pdf
import pyecharts.options as opts
from copy import copy
from pyecharts.charts import Bar, Line, Page


class EdaTools:

    @staticmethod
    def df_type_transform(df: pd.DataFrame, df_types: dict) -> pd.DataFrame:
        """
        对数据集进行变量转换,目前支持类型如下:
            - int
            - float
            - category/str
        df_types example: {"a": "int", "b":"float", "c":"category"}
        :param df: 数据集
        :param df_types: 变量对应的数据类型
        :return:
        """
        for k in df_types.keys():
            if k not in df.columns:
                logging.warning("  feature: '{}' not in you dataset please check you df_types".format(k))
                continue
            try:
                df[k] = df[k].astype(df_types[k])
            except ValueError:
                logging.warning("feature: {} dtype incorrect!".format(k))
                continue
        return df

    @staticmethod
    def var_describe(df: pd.DataFrame, var: str) -> dict:
        """
        变量统计描述: 总数,平均数,方差,极值,分位数
        :param df:
        :param var:
        :return:
        """
        desc = df[var].describe()
        return dict(desc)

    @staticmethod
    def missing_rate(df: pd.DataFrame, var: str) -> dict:
        """
        计算变量缺失率
        :param df:
        :param var:
        :return:
        """
        missing = len(df) - df[var].count()
        missing_rate = round(missing / len(df), 4)
        return {"missing_rate": missing_rate}

    @staticmethod
    def unique_value(df: pd.DataFrame, var: str) -> dict:
        """
        计算变量的unique_value的个数
        :param df:
        :param var:
        :return:
        """
        unique_num = len(df[var].unique())
        return {"unique_num": unique_num}

    @staticmethod
    def name_desc(desc_ch: str) -> dict:
        """
        为变量添加中文描述
        :param desc_ch:
        :return:
        """
        return {"desc_ch": desc_ch}

    @staticmethod
    def replace_dirty_null(df: pd.DataFrame, var: str):
        """
        空值标准化
        :param df:
        :param var:
        :return:
        """
        return df[var].replace([' ', '', 'null', 'Null', 'NULL', 'nan', 'np.nan', 'Nan', 'NaN', 'NAN'], np.nan)

    @staticmethod
    def risk_distinction(df: pd.DataFrame, var: str) -> dict:
        """
        对一些缺失率高/重复值占比高但是一旦有值/值不相同badrate就会非常高的变量进行mark
        :param df:
        :param var:
        :return:
        """
        pass

    @staticmethod
    def plot_density(df: pd.DataFrame, var: str, y: str, img_path=None, show=False) -> None:
        """
        绘制变量的概率密度曲线
        # 变量概率密度
        # os.mkdir(report_dir + "/" + img_path)
        # EdaTools.plot_density(df, var, y, img_path=img_path, show=show)
        :param df:
        :param var:
        :param y:
        :param img_path:
        :return:
        """
        data = df[[var, y]]
        if len(df[y].unique()) > 2:
            warnings.warn("Y categories > 2, check you dataset?")

        for i in df[y].unique():
            sns.distplot(data[data[y] == i][var])
        if img_path is not None:
            plt.savefig(img_path + "{}.png".format(var))
        if show:
            plt.show()

    @staticmethod
    def corr_filter(df: pd.DataFrame, threshold=0.9):
        df_corr = df.corr().apply(abs)

        # 对角线替换为na ==> fillna(0)
        for i in range(len(df_corr.columns)):
            df_corr.iloc[i, i] = None
        df_corr = df_corr.fillna(0)

        # 筛选
        result = {}
        for i in df_corr.index:
            for j in df_corr.columns:
                if df_corr.loc[i][j] >= threshold:
                    if f"{j}~{i}" in result.keys():
                        pass
                    else:
                        result[f"{i}~{j}"] = df_corr.loc[i][j]

        corr_weight = []
        for k, v in result.items():
            var1, var2 = str(k).split("~")
            corr_weight.append({
                "var": var1,
                "weight": v
            })
            corr_weight.append({
                "var": var2,
                "weight": v
            })
        corr_weight = pd.DataFrame(corr_weight)
        corr_res = corr_weight.groupby(by="var").apply(lambda x: x["weight"].sum())
        corr_res = corr_res.sort_values(ascending=False)

        final_drop = []
        result_new = copy(result)
        for v in corr_res.index:
            if len(result_new) == 0:
                break
            else:
                final_drop.append(v)
                current_key = copy(list(result_new.keys()))
                for k in current_key:
                    if str(k).find(v) != -1:
                        result_new.pop(k)
        return final_drop

    @staticmethod
    def profile_filter_corr(report: pdf.ProfileReport, threshold: float):
        """
        pandas_profile对应的删除高相关性变量的方法:
            - 两个相关性高的变量,删除缺失高的那个
        :param report: pandas_profile.ProfileReport
        :param threshold: 阈值
        :return: 高相关变量列表
        """
        high_corr_var = report.get_rejected_variables(threshold=threshold)
        return high_corr_var

    @staticmethod
    def profile_filter_missing(report: pdf.ProfileReport, threshold: float):
        """
        pandas_profile对应的删除高缺失变量的方法
        :param report: pandas_profile.ProfileReport
        :param threshold: 阈值
        :return: 高缺失变量列表
        """
        df_distribute = report.get_description().get('variables')
        high_miss_var = list(df_distribute.ix[df_distribute['p_missing'] >= threshold].index)
        return high_miss_var

    @staticmethod
    def profile_filter_spar(report: pdf.ProfileReport, threshold: float):
        """
        pandas_profile对应的稀疏比例高的变量
        :param report:
        :param threshold:
        :return:
        """
        df_distribute = report.get_description().get('variables')
        high_spar = list(df_distribute.ix[df_distribute['p_zeros'] >= threshold].index)
        return high_spar

    @staticmethod
    def profile_filter_uniq(report: pdf.ProfileReport, threshold: float):
        """
        @author: jiangguixiang
        pandas_profile对应的类别型而且基本是一对一的变量：身份证类，号码类
        :param report:
        :param threshold:
        :return:
        """
        df_distribute = report.get_description().get('variables')
        high_spar = list(df_distribute.ix[(df_distribute['p_unique'] >= threshold) & (df_distribute.type == 'CAT')].index)
        return high_spar

    @staticmethod
    def corr_analysis(df: pd.DataFrame, threshold: float=0.9):
        """
        相关性分析
        :param df:
        :param threshold:
        :return:
        """
        df_corr = df.corr().apply(abs)
        df_corr_filtered = (df_corr >= threshold)
        result = []
        for i in df_corr_filtered.index:
            for c in df_corr_filtered.columns:
                if (str(i) != str(c)) and df_corr_filtered.loc[i, c]:
                    result.append(f"{i}~{c}")
        return result

    @staticmethod
    def plot_fluctuation_by_time(df: pd.DataFrame,
                                 var: str,
                                 y: str,
                                 time_column: str,
                                 filename: str=None,
                                 stats_methods=None):
        """
        绘制一个变量按照时间波动的bar
        :param df:
        :param var:
        :param y:
        :param time_column:
        :param stats_methods:
        :return:
        """
        sub_df = df[[var, y, time_column]].copy()
        result = []
        for sub in sub_df.groupby(by=time_column):
            tmp = {}
            t = sub[0]
            t_df = pd.DataFrame(sub[1])
            tmp[time_column] = t
            tmp["missing"] = EdaTools.missing_rate(t_df, var)["missing_rate"]
            desc = EdaTools.var_describe(t_df, var)
            tmp.update(desc)
            tmp["unique_num"] = EdaTools.unique_value(t_df, var)["unique_num"]
            result.append(tmp)
        result = pd.DataFrame(result).sort_values(by=time_column, ascending=True)
        x = time_column
        y = list(result.drop(time_column, axis=1).columns)

        page = Page(layout=Page.SimplePageLayout)
        for i in y:
            l = Line()\
                .add_xaxis(list(result[x]))\
                .add_yaxis(i, list(result[i]))\
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="{}:{}".format(var, i), pos_left="left"),
                    legend_opts=opts.LegendOpts(type_="scroll", pos_top="7%"),
                )\
                .add_js_funcs("document.getElementsByTagName('body')[0].style.zoom=0.67")
            # try:
            #     l.set_global_opts(yaxis_opts = opts.AxisOpts(min_=min(list(result[i])) * 0.90))
            # except:
            #     pass
            page.add(copy(l))
        if filename:
            page.render(filename)
        return page


def eda_report_df(df: pd.DataFrame, df_types: dict, y: str, report_dir: str):
    """
    eda报告, 在生成eda报告之前需要删除数据集中的无用列
    该方法会对数据集进行数据类型的转换,返回转换好的数据集具体转换方法见: EdaTools.df_type_transform
    :param df: 数据集
    :param df_types: 字段对应数据类型
    :param y:
    :param report_dir:
    :return:
    """
    df = EdaTools.df_type_transform(df, df_types)
    columns = df.drop(y, axis=1).columns
    report = []
    for var in columns:
        tmp = {}
        # 变量名
        tmp["var_name"] = var
        # 变量描述
        desc = EdaTools.var_describe(df, var)
        tmp.update(desc)
        # 变量缺失率
        missing_rate = EdaTools.missing_rate(df, var)
        tmp.update(missing_rate)
        # 变量唯一值
        unique_num = EdaTools.unique_value(df, var)
        tmp.update(unique_num)
        # 数据整合
        report.append(copy(tmp))
    # 将eda报告进行保存
    pd.DataFrame(report).to_csv(report_dir + "eda_report.csv", index=False, encoding="utf8")
    return df


def eda_report_html(df: pd.DataFrame, var_types: dict, report_path=None):
    """
    通过pandas_profile来出eda报告
    :param df: 数据集
    :param var_types: 变量名: 变量类型
    :param report_dir:
    :return:
    """
    multiprocessing.freeze_support()
    df_new = df.copy()
    df_new = EdaTools.df_type_transform(df_new, var_types)
    for c in df.columns:
        df_new[c] = EdaTools.replace_dirty_null(df, c)
    report = pdf.ProfileReport(df_new)
    if report_path:
        report.to_file(report_path)
    return report
