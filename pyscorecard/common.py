# -*- coding: utf8 -*-
from .Tool import Bins
from .plot.plot_bin import BinPlot


def bin_plot_watch(df, df_test, df_oot, var, y, filepath, bin_num=20):
    """
    卡方,等频分箱bin对比
    :param df: 数据集
    :param var: 变量列表
    :param y: y
    :param filepath: 文件名
    :param bin_num: 等频分箱的箱数
    :return: 卡方/等频的对比图html
    """
    bb = Bins()
    res = bb.generate_raw(df[var], df[y])
    all_report, change_report, woe_df, bin_df, false_dict = res
    all_report["Interval"] = all_report["Interval"].apply(lambda x: str(x))
    bp = BinPlot(
        df=df,
        all_report=all_report,
        change_report=change_report,
        y=y,
        df_test=df_test,
        df_oot=df_oot,
        bin_num=bin_num
    )
    bp.plot_bin(all_report, filepath)


def bin_trend_plot(df, var, y):
    """
    变量长期趋势
    :param df:
    :param var:
    :param y:
    :return:
    """
    raise NotImplementedError
    # # 生成长期趋势图
    # cur_selected = change_report["Feature"]
    # btp = BinTrendPlot(self.train, cur_selected, self.y, self.time_column, change_report)
    # btp.plot_bin(f"{report_dir}/bin_trend_plot_{self.bin_version}.html")
