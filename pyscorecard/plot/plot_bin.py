# -*- coding: utf8 -*-
"""
画图
"""
import pandas as pd
import pyecharts
import pyecharts.options as opts
import os
from copy import copy
from ..Tool import Bins
from ..binning import Binning
from pyecharts.charts import Bar, Line, Grid, Page
from pyecharts.components import Table
from jinja2 import Environment, FileSystemLoader
from pyecharts.globals import CurrentConfig
from pyecharts.render.engine import RenderEngine
from pyecharts.commons.utils import write_utf8_html_file
from ..tools import interval_round, calc_woe, calc_iv


# print(os.path.abspath(os.path.dirname(__file__)) + "/template")
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 2500)


class BinPlot:
    """
    左: 卡方分箱图 | 右: 等频分箱图
    左: 卡方分箱数据 | 右: 等频分箱数据
    目前版本封装了很多功能,不是很灵活,后续会考虑改善
    todo: 后续版本的pyecharts会更新page的dashboard功能,更新之后第一时间改写这部分(网页排版问题)
    """
    def __init__(self, df: pd.DataFrame, all_report: pd.DataFrame, change_report: pd.DataFrame, y: str, df_test: pd.DataFrame, df_oot: pd.DataFrame, bin_num=20):
        """

        :param df: 数据集
        :param all_report: Tool.Bins 生成的bin的report
        :param y: y
        """
        bb = Bins()
        self.bin_num = bin_num
        self.df = df
        self.bindf_chi = all_report
        self.bindf_chi_test = bb.mannual_rebin(df_test, change_report, df_test[y])
        self.bindf_chi_oot = bb.mannual_rebin(df_oot, change_report, df_oot[y])
        self.y = y
        self.bindf_freq = self.generate_freq_bin_report()
        template_path = os.path.abspath(os.path.dirname(__file__)) + "/template"
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader(template_path))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()

        # self.dtype_dict = self.get_type_dict(change_report)
        # self.change_report = change_report

    def get_type_dict(self, change_report: pd.DataFrame):
        res = {}
        for i in range(len(change_report)):
            res[change_report.loc[i]["Feature"]] = {
                "cut": change_report.loc[i]["Interval"],
                "type": change_report.loc[i]["type"]
            }
        return res

    def generate_sub_all_report(self, subdf: pd.DataFrame, var: str, y: str, cuts):
        """
        # demo = Bins()
        # # 启动运算
        # demo.generate_bin_smi(x, y, ftype='C')
        # # 输出分箱报告
        # bin_stat, bin_interval, bin_map = demo.get_bin_info()
        # # 按照这种分箱将变量替换为箱子值
        # bin_result = demo.value_to_bin(x)
        # # 按照分箱替换的结果计算woe等
        # woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
        # # 按照woe结果将新数据进行woe替换
        # woe_result, woe_report, error_flag = Bins.woe_iv(
        #     bin_result, report=woe_report)
        :param subdf:
        :param var:
        :param y:
        :param cuts:
        :return:
        """
        bb = Bins()
        bb.generate_bin_smi(
            subdf[var],
            subdf[y],
            interval=cuts,
            ftype=self.dtype_dict[var]
        )
        bin_stat, bin_interval, bin_map = bb.get_bin_info()
        bin_result = bb.value_to_bin(subdf[var])
        woe_result, woe_report, error_flag = bb.woe_iv(bin_result, subdf[y])
        woe_report["Feature"] = var
        report = pd.merge(bin_stat, woe_report, left_on="Bin", right_on="Bin")
        return report

    def plot_var_trend(self, df: pd.DataFrame, var: str, y: str, time_column: str):
        this_df = df[[var, y, time_column]]
        this_df_grouped = this_df\
            .sort_values(by=time_column)\
            .groupby(by=time_column)
        cuts = self.dtype_dict[var]["cut"]
        result = []

        for sub in this_df_grouped:
            sub_report = self.generate_sub_all_report(sub[1], var, y, cuts)
            sub_report["dt"] = sub[0]
            sub_report = sub_report.sort_values(by="Bin")
            sub_report.index = sub_report["Bin"]

            tmp = {}
            tmp["dt"] = sub[0]
            tmp["iv"] = sub_report.iloc[0]["iv"]
            for b in sub_report["Bin"]:
                tmp[f"{b}_badrate"] = sub_report.loc[b]["PD"]
                tmp[f"{b}_total_prop"] = sub_report.loc[b]["total_prop"]
            result.append(tmp)
        result = pd.DataFrame(result)

        # 画图: IV
        line_iv = Line()\
            .add_xaxis(list(result["dt"]))\
            .add_yaxis("IV", list(result["iv"]))\
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{var}_IV_Trend"),
                datazoom_opts=[opts.DataZoomOpts(is_show=True), ],
            )

        # 画图: TotalProp
        bar_prop = Bar().add_xaxis(list(result["dt"]))
        for b in sub_report["Bin"]:
            this_interval = sub_report.loc[b]["Interval"]
            bar_prop.add_yaxis(this_interval, list(result[f"{b}_total_prop"]), stack=True)
        bar_prop.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{var}_Bin_prop_Trend"),
            datazoom_opts=[opts.DataZoomOpts(is_show=True), ],
        )

        # 画图: BadRate
        line_badrate = Line().add_xaxis(list(result["dt"]))
        for b in sub_report["Bin"]:
            this_interval = sub_report.loc[b]["Interval"]
            line_badrate.add_yaxis(this_interval, list(result[f"{b}_badrate"]), stack=True)
        line_badrate.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{var}_Bin_BadRate_Trend"),
            datazoom_opts=[opts.DataZoomOpts(is_show=True), ],
        )
        return {
            "trend_iv": self.prepare_render(line_iv),
            "trend_prop": self.prepare_render(bar_prop),
            "trend_badrate": self.prepare_render(line_badrate)
        }

    def generate_freq_bin_report(self):
        """
        生成等频分箱的report
        :return:
        """
        bb = Bins()
        var_type_map = self.bindf_chi[["Feature", "type"]]\
            .drop_duplicates(subset=["Feature", "type"])\
            .reset_index()
        var_type_dict = {k: v for k, v in zip(var_type_map["Feature"], var_type_map["type"])}
        bindf_freq = []
        for sub in var_type_dict.items():
            this_cut = Binning.binning_frequency(self.df, sub[0], self.bin_num, sub[1])
            # print(this_cut)
            bb.generate_bin_smi(
                self.df[sub[0]],
                self.df[self.y],
                interval=this_cut,
                ftype=sub[1]
            )
            this_df = bb.get_bin_info()[0]
            this_df["Feature"] = sub[0]
            this_df["Interval"] = this_df["Interval"].apply(interval_round)
            this_df["PD"] = this_df["PD"].round(4)
            this_df["1_prop"] = this_df["1_prop"].round(4)
            this_df["0_prop"] = this_df["0_prop"].round(4)
            this_df["total_prop"] = this_df["total_prop"].round(4)
            this_df["PD"] = this_df["PD"].round(4)

            bindf_freq.append(this_df.copy())
            # print(list(this_df.columns))
        bindf_freq = pd.concat(bindf_freq, axis=0)
        return bindf_freq

    def prepare_render(self, chart):
        """
        pyechart prepare_render
        :param chart:
        :return:
        """
        if isinstance(chart, pyecharts.components.Table):
            return chart
        else:
            chart._prepare_render()
            return self.render_engine.generate_js_link(chart)

    def render_template(self, template_name, filename, **kwargs):
        """
        自定义渲染方法
        :param template_name:
        :param kwargs:
        :param filename:
        :return:
        """
        tpl = self.render_engine.env.get_template(template_name)
        html = tpl.render(**kwargs)
        write_utf8_html_file(filename, self.render_engine._replace_html(html))

    def plot_bin_detail(self, bindf: pd.DataFrame, title=""):
        # todo: 这个地方的实现太丑了,后面需要看源码修改一下
        ###########

        ###########
        bindf = bindf.sort_values(by="Bin")
        if "woe" in bindf.columns:
            try:
                bindf = bindf[[
                    'Feature',
                    'Interval',
                    'total',
                    'total_prop',
                    '0',
                    '1',
                    "0_prop",
                    "1_prop",
                    'PD',
                    'woe',
                    'iv',
                ]]
            except KeyError:
                bindf = bindf[[
                    'Feature',
                    'Interval',
                    'total',
                    'total_prop',
                    0,
                    1,
                    "0_prop",
                    "1_prop",
                    'PD',
                    'woe',
                    'iv',
                ]]
            bindf["woe"] = bindf["woe"].round(4)
            bindf["iv"] = bindf["iv"].round(4)
        else:
            print(bindf.columns)
            try:
                bindf = bindf[[
                    'Feature',
                    'Interval',
                    'total',
                    'total_prop'
                    '0',
                    '1',
                    "0_prop",
                    "1_prop",
                    "PD",
                ]]
            except KeyError:
                bindf = bindf[[
                    'Feature',
                    'Interval',
                    'total',
                    'total_prop',
                    0,
                    1,
                    "0_prop",
                    "1_prop",
                    "PD",
                ]]
            bindf["woe"] = bindf.apply(calc_woe, axis=1)
            ivcs = bindf.apply(calc_iv, axis=1)
            bindf["iv"] = round(ivcs.sum(), 4)

        # bindf["total_pct"] = round(bindf["total"] / bindf["total"].sum(), 4)
        bindf["total_prop"] = bindf["total_prop"].round(4)
        bindf["1_prop"] = bindf["1_prop"].round(4)
        bindf["0_prop"] = bindf["0_prop"].round(4)
        bindf["PD"] = bindf["PD"].round(4)
        table = Table()
        headers = list(bindf.columns)
        rows = [list(bindf.iloc[i]) for i in range(len(bindf))]
        table.add(headers, rows)
        table.set_global_opts(opts.ComponentTitleOpts(title=title, subtitle=""))
        return table

    def _plot_bin_badrate(self, df: pd.DataFrame, title, title_pos, width):
        line = Line()
        df["Interval"] = df["Interval"].apply(lambda x: str(x))
        if df["type"][0] == "C":
            df = df.sort_values(by="Bin")
        # print(df)
        # print("BBBBBBBBBBBBBBBB")
        x = list(str(x) for x in df["Interval"])
        line.add_xaxis(x)
        df["PD"] = df["PD"].round(3)
        df["total_prop"] = df["total_prop"].round(3)
        line.add_yaxis("badRate", list(df["PD"]), is_selected=True)
        # line.add_yaxis("total_prop", list(df["total_prop"]), is_selected=False)
        line.set_global_opts(title_opts=opts.TitleOpts(title=title, **title_pos),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-60))
                             )
        bar = Bar(init_opts=opts.InitOpts(width=width))\
            .add_xaxis(x)\
            .add_yaxis("total_prop", list(df["total_prop"]), color="#fab27b", itemstyle_opts=opts.ItemStyleOpts(color="#60ACFC"))
        line.overlap(bar)
        return self.prepare_render(line)

    def plot_bin_compare(self, varname: str, bindf_chi: pd.DataFrame, bindf_freq: pd.DataFrame):
        """
        两个分箱对比
        :param bindf_chi:
        :param bindf_freq:
        :return: grid
        """
        c_chi = self._plot_bin_badrate(bindf_chi, "{}: 卡方分箱".format(varname), {"pos_left": "10%"}, width="66%")
        c_freq = self._plot_bin_badrate(bindf_freq, "{}: 等频分箱".format(varname), {"pos_right": "10%"}, width="33%")
        grid = Grid(init_opts=opts.InitOpts(width="1200px")) \
            .add(c_chi, grid_opts=opts.GridOpts(pos_right="65%", pos_top="100px"))\
            .add(c_freq, grid_opts=opts.GridOpts(pos_left="45%", pos_top="100px"))

        return self.prepare_render(grid)

    def plot_badrate_diff_data(self, sub_tr, sub_test, sub_oot):
        train_report = sub_tr.sort_values(by="Bin")
        train_report["PD"] = train_report["PD"].round(3)
        train_report["total_prop"] = train_report["total_prop"].round(3)
        test_report = sub_test.sort_values(by="Bin")
        test_report["PD"] = test_report["PD"].round(3)
        test_report["total_prop"] = test_report["total_prop"].round(3)
        oot_report = sub_oot.sort_values(by="Bin")
        oot_report["PD"] = oot_report["PD"].round(3)
        oot_report["total_prop"] = oot_report["total_prop"].round(3)

        line1 = Line()\
            .add_xaxis(list(train_report["Interval"]))\
            .add_yaxis("训练集badrate", list(train_report["PD"]), itemstyle_opts=opts.ItemStyleOpts(color="#60ACFC"))\
            .add_yaxis("测试集badrate", list(test_report["PD"]), itemstyle_opts=opts.ItemStyleOpts(color="#60ACFC")) \
            .add_yaxis("时间外样本badrate", list(oot_report["PD"]), itemstyle_opts=opts.ItemStyleOpts(color="#60ACFC"))\
            .set_global_opts(title_opts=opts.TitleOpts(title="BadRate: train|test|oot对比"),
                             xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-25)),
                             yaxis_opts=opts.AxisOpts(name="BadRate", name_gap=30, name_location="center"),
                             legend_opts=opts.LegendOpts(pos_top="4%"),)\
            .extend_axis(yaxis=opts.AxisOpts(name="Total Prop", name_location="center", axislabel_opts=opts.LabelOpts(formatter="{value}")))

        bar1 = Bar() \
            .add_xaxis(list(train_report["Interval"])) \
            .add_yaxis("训练集total_prop", list(train_report["total_prop"]), yaxis_index=1, category_gap="40%", gap='5%') \
            .add_yaxis("测试集total_prop", list(test_report["total_prop"]),  yaxis_index=1, category_gap="40%", gap='5%') \
            .add_yaxis("时间外样本total_prop", list(oot_report["total_prop"]),  yaxis_index=1, category_gap="40%", gap='5%')
        line1.overlap(bar1)
        return self.prepare_render(line1)

    def plot_bin(self, all_report: pd.DataFrame, filename: str, time_column=None):
        """
        默认按照Bins生成的all_report的格式来构造数据,也可以自己构造,只需要有下面5列

            'Feature',
            'Interval',
            '0',
            '1',
            'total'
        :param all_report:
        :return:
        """
        info = []
        for sub in all_report.groupby(by="Feature"):
            try:
                sub_bindf_freq = self.bindf_freq[self.bindf_freq["Feature"] == sub[0]]
                sub_test = self.bindf_chi_test[self.bindf_chi_test["Feature"] == sub[0]]
                sub_oot = self.bindf_chi_oot[self.bindf_chi_oot["Feature"] == sub[0]]
            except Exception as e:
                print(e)
                continue
            tmp = {}
            tmp["varname"] = sub[0]
            sub_df = pd.DataFrame(sub[1]).sort_values(by="Interval")
            tmp["chart_badrate"] = self.plot_bin_compare(sub[0], sub_df, sub_bindf_freq)
            tmp["chart_diff"] = self.plot_badrate_diff_data(sub_df, sub_test, sub_oot)
            tmp["chart_detail"] = self.plot_bin_detail(sub_df, title=f"{sub[0]}:卡方分箱详细数据")
            tmp["chart_detail_freq"] = self.plot_bin_detail(sub_bindf_freq, title=f"{sub[0]}: 等频分箱详细数据")
            info.append(copy(tmp))
        self.render_template("bin_plot.html", filename, info=info)
