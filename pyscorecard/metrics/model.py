# -*- coding: utf8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pyecharts.options as opts
from copy import copy
from sklearn.metrics import roc_auc_score, auc, roc_curve
from ..model import mylogit
from pyecharts.charts import Line, HeatMap, Bar
from ..plot.lift_chart import LiftChart
from ..binning import BinningTools


pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 2500)


def lift_ks(y_test, score, bins=20, out_dir=None, save_label=None):
    """
    @author: zhengbifan zkj
    :param y_test:
    :param score:
    :param bins:
    :return:
    """
    df = pd.DataFrame({'y_test': y_test, 'score': score})
    df = df.sort_values('score')
    grouping = pd.qcut(df.score, bins, duplicates='drop')
    # grouped = df['y_test'].groupby(grouping)
    # df = grouped.apply(lambda x: {'count': x.count(), 'bad': x.sum(), 'pred': x.mean()}).unstack()

    grouped = df.groupby(grouping)
    df = grouped.apply(lambda x: {'count': x["y_test"].count(), 'bad': x["y_test"].sum(), 'pred': x["score"].mean()})
    df = pd.DataFrame(list(df), index=df.index)
    df = df.reset_index()
    df = df.sort_index(ascending=True)
    df['good'] = df['count'] - df['bad']
    df['bin_rate'] = df['count'] / sum(df['count'])
    df['bad_per_cumsum'] = df['bad'].cumsum() / sum(df['bad'])
    df['good_per_cumsum'] = df['good'].cumsum() / sum(df['good'])
    df['bad_rate'] = df['bad'] / df['count']
    df['bad_rate_cumsum'] = df['bad'].cumsum() / df['count'].cumsum()
    df['ks_score'] = abs(df['bad_per_cumsum'] - df['good_per_cumsum'])

    # 控制一下小数位数
    df["bin_rate"] = df["bin_rate"].round(4)
    df["bad_per_cumsum"] = df["bad_per_cumsum"].round(4)
    df["good_per_cumsum"] = df["good_per_cumsum"].round(4)
    df["bad_rate"] = df["bad_rate"].round(4)
    df["bad_rate_cumsum"] = df["bad_rate_cumsum"].round(4)
    df["ks_score"] = df["ks_score"].round(4)
    df["pred"] = df["pred"].round(4)

    if out_dir and save_label:
        lc = LiftChart(df)
        lc.plot(save_label=save_label, out_dir=out_dir)
    return df


def auc_score(y_true: list, y_pred: list):
    """
    计算模型auc
    :param y_true:
    :param y_pred:
    :return: auc score
    """
    auc = roc_auc_score(y_true, y_pred, average='macro', sample_weight=None)
    return auc


def ks_score(y_true: list, y_pred: list):
    fpr_train, tpr_train, thresholds = roc_curve(list(y_true), list(y_pred))
    ks = max(tpr_train - fpr_train)
    return ks


def roc_plot(y_true, y_pred, save_label, result_path):
    """
    # 绘制ROC曲线图
    @author liqian
    : param y_true : pd.series,标签列
    : param y_pred : pd.series,预测分值列
    : param save_label : string,保存图片前缀名
    : param result_path : string，保存文件的路径
    : return roc_auc ：float，auc值
    : return fig ：roc曲线
    """
    fpr_train, tpr_train, thresholds = roc_curve(list(y_true),list(y_pred))
    roc_auc = auc(fpr_train, tpr_train)
    fig = plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_train, tpr_train, 'b', label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # result_path = os.path.join(result_path, 'figure/AUC/')
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    plt.savefig(os.path.join(result_path, save_label + '_AUC.png'), format='png', dpi=80)
    return roc_auc, fig


def ks_plot(y_true, y_pred, save_label, result_path):
    """
    # 绘制KS曲线图
    : param y_true : pd.series,标签列
    : param y_pred : pd.series,预测分值列
    : param save_label : string,保存图片前缀名
    : param result_path : string，保存文件的路径
    : return KS ：float，KS值
    : return fig ：KS曲线
    """
    fpr_train, tpr_train, thresholds = roc_curve(list(y_true), list(y_pred))
    ks_score = max(tpr_train - fpr_train)
    fig = plt.figure(figsize=(8, 6))
    plt.title('KS')
    plt.plot(tpr_train, 'b', label='true_positive')
    plt.plot(fpr_train, 'r', label="false_positive")
    plt.plot(tpr_train - fpr_train, 'g', label='KS = %0.3f'% ks_score)
    plt.legend(loc='lower right')
    if save_label and result_path:
        plt.savefig(os.path.join(result_path, save_label + '_AUC.png'), format='png', dpi=80)
    return ks_score, fig


def accum_auc(model_result, df: pd.DataFrame, y: str, filepath: str):
    """
    累计auc
    根据变量重要性排序,逐渐加变量,计算auc累计值
    :param varlist_by_importance:
    :param df:
    :param y:
    :param filepath:
    :return:
    """
    varlist_by_importance = list(model_result.pvalues.sort_values().keys())
    if "const" in varlist_by_importance:
        varlist_by_importance.remove("const")
    accum_auc_value = []
    for i in range(len(varlist_by_importance)):
        this_x = varlist_by_importance[: i + 1]
        logit_result = mylogit(x=df[this_x], y=df[y], add_constant=True, select=None)
        y_pred = logit_result.predict(sm.add_constant(df[this_x]))
        y_true = df[y]
        this_auc = round(roc_auc_score(y_true, y_pred), 4)
        accum_auc_value.append(this_auc)
    line = Line() \
        .add_xaxis(varlist_by_importance) \
        .add_yaxis("variable", accum_auc_value) \
        .set_global_opts(
            title_opts=opts.TitleOpts(title="累计AUC图",
                                      subtitle="按照变量重要性,逐渐加变量"),
            xaxis_opts=opts.AxisOpts(name="变量",
                                     name_location="end",
                                     name_gap="30",
                                     axislabel_opts=opts.LabelOpts(is_show=True, rotate=-30)),
            yaxis_opts=opts.AxisOpts(name="累计AUC",
                                     name_location="center",
                                     name_gap="30",
                                     min_=min(accum_auc_value)*0.95))
    if filepath:
        line.render(filepath)
    return line


def predict_vs_actual(var_value: list, dtype: str, y_true: list, y_pred: list, cuts: list):
    """
    predict_vs_actual
    var_value, y_true, y_pred 一定要保持index一致!
    :param var_value: 变量实际值
    :param y_true: 真实的y
    :param y_pred: y预测值
    :param cuts: 分箱点
    :return:
    """
    df = pd.DataFrame({
        "x_value": var_value,
        "y_true": y_true,
        "y_pred": y_pred
    })
    print(cuts, type(cuts))
    if dtype == "num":
        cuts = [float("-inf"), ] + cuts + [float("+inf"), ]
        df["cuts"] = pd.cut(df["x_value"], bins=cuts, right=False)
    elif dtype == "char":
        df["cuts"] = BinningTools.cut_str(df["x_value"], bins=cuts)
    result = []
    total = len(df)
    for sub in df.groupby(by="cuts"):
        tmp = {}
        this_cut = str(sub[0])
        this_df = sub[1]
        tmp["cuts"] = this_cut
        try:
            tmp["badrate"] = len(this_df[this_df["y_true"] == 1]) / len(this_df)
        except ZeroDivisionError:
            tmp["badrate"] = 0
        tmp["y_pred_mean"] = this_df["y_pred"].mean()
        tmp["total_prop"] = len(this_df) / total
        result.append(copy(tmp))

    result = pd.DataFrame(result)
    result = result.sort_values(by="cuts")
    # 约束一下小数点
    result["badrate"] = result["badrate"].round(4)
    result["y_pred_mean"] = result["y_pred_mean"].round(4)
    result["total_prop"] = result["total_prop"].round(4)

    line1 = Line(init_opts=opts.InitOpts(width="400px"))\
        .add_xaxis(list(result["cuts"]))\
        .add_yaxis("badrate", list(result["badrate"]))\
        .set_global_opts(title_opts=opts.TitleOpts(title="Predict vs Actual"),
                         legend_opts=opts.LegendOpts(pos_top="40px")) \
        .extend_axis(yaxis=opts.AxisOpts(name="Total Prop",
                                         name_location="center",
                                         axislabel_opts=opts.LabelOpts(formatter="{value}")))

    line2 = Line(init_opts=opts.InitOpts(width="400px")) \
        .add_xaxis(list(result["cuts"])) \
        .add_yaxis("y_pred_mean", list(result["y_pred_mean"])) \
        .set_series_opts(label_opts=opts.LabelOpts(position="bottom"),
                         linestyle_opts=opts.LineStyleOpts(type_="dotted")
                         ) \
        .set_global_opts(legend_opts=opts.LegendOpts(pos_top="40px"))

    bar1 = Bar(init_opts=opts.InitOpts(width="400px"))\
        .add_xaxis(list(result["cuts"]))\
        .add_yaxis("TotalProp", list(result["total_prop"]), yaxis_index=1)

    line1.overlap(line2)
    line1.overlap(bar1)
    return line1


def corr_heat_map(df: pd.DataFrame, cols: list):
    """
    变量相关性的热力图
    建议少于20个变量,多了看不清
    :param df:
    :param cols:
    :return:
    """
    corr = df[cols].corr(method="pearson")
    corr = corr.round(3)
    corr = corr.apply(lambda x: abs(x))
    myvalues = []
    for i in range(len(corr.index)):
        for j in range(len(corr.index)):
            tmp = [i, j, corr.iloc[i, j]]
            myvalues.append(copy(tmp))
    heat_map = HeatMap(init_opts=opts.InitOpts(width=1200))\
        .add_xaxis(list(corr.columns))\
        .add_yaxis("corr", list(corr.index), myvalues) \
        .set_global_opts(
            title_opts=opts.TitleOpts(title="模型变量相关性"),
            datazoom_opts=[opts.DataZoomOpts(is_show=True, is_realtime=True), ],
            visualmap_opts=opts.VisualMapOpts(min_=-1.2, max_=1.2, pos_right=20),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", is_scale=True, is_inverse=True, axislabel_opts=opts.LabelOpts(is_show=True, rotate=-60)),
            yaxis_opts=opts.AxisOpts(is_scale=True, is_inverse=False, axislabel_opts=opts.LabelOpts(is_show=True, position="right")),
            tooltip_opts=opts.TooltipOpts(is_show=True))\
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="insideBottom"))
    return heat_map
