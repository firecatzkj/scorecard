# -*- coding: utf8 -*-
"""
画图
"""
import pandas as pd
import pyecharts
import pyecharts.options as opts
import os
import json
from copy import copy
from ..Tool import Bins
from ..binning import Binning
from pyecharts.charts import Bar, Line, Grid, Page
from pyecharts.components import Table
from jinja2 import Environment, FileSystemLoader
from pyecharts.globals import CurrentConfig
from pyecharts.render.engine import RenderEngine
from pyecharts.commons.utils import write_utf8_html_file


pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 2500)


class BinTrendPlot:

    def __init__(self, df: pd.DataFrame, var_list: list, y: str, time_column: str, change_report: pd.DataFrame):
        template_path = os.path.abspath(os.path.dirname(__file__)) + "/template"
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader(template_path))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()
        self.df = df
        self.var_list = var_list
        self.y = y
        self.time_column = time_column
        self.change_report = change_report.set_index("Feature")

    def generate_sub_all_report(self, subdf: pd.DataFrame, var: str, y: str, cuts, ftype):
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
            ftype=ftype
        )
        bin_stat, bin_interval, bin_map = bb.get_bin_info()
        bin_result = bb.value_to_bin(subdf[var])
        woe_result, woe_report, error_flag = bb.woe_iv(bin_result, subdf[y])
        woe_report["Feature"] = var
        report = pd.merge(bin_stat, woe_report, left_on="Bin", right_on="Bin")
        return report

    def plot_var_trend(self, df: pd.DataFrame, var: str, y: str, time_column: str):
        this_df = df[[var, y, time_column]]
        # 类型强转, 避免画图缺失
        this_df[time_column] = this_df[time_column].astype(str)
        this_df_grouped = this_df\
            .sort_values(by=time_column)\
            .groupby(by=time_column)

        cuts = json.loads(self.change_report.loc[var]["Interval"])
        # print(var, type(cuts), "LLLLLLLLLLLLLLLLLLLLL", cuts)
        ftype = self.change_report.loc[var]["type"]
        result = []
        for sub in this_df_grouped:
            sub_report = self.generate_sub_all_report(sub[1], var, y, cuts, ftype)
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
        result = pd.DataFrame(result).round(4)

        # 画图: IV
        print(result)
        line_iv = Line()\
            .add_xaxis(list(result["dt"]))\
            .add_yaxis("IV", list(result["iv"]))\
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{var}_IV_Trend"),
                datazoom_opts=[opts.DataZoomOpts(is_show=True, range_start=0, range_end=100), ],
            )

        # 画图: TotalProp
        bar_prop = Bar().add_xaxis(list(result["dt"]))
        for b in sub_report["Bin"]:
            this_interval = str(sub_report.loc[b]["Interval"])
            bar_prop.add_yaxis(this_interval, list(result[f"{b}_total_prop"]), stack=True, category_gap=1)
        bar_prop.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{var}_Bin_prop_Trend"),
            datazoom_opts=[opts.DataZoomOpts(is_show=True, range_start=0, range_end=100), ],
            legend_opts=opts.LegendOpts(is_show=True, orient="vertical", pos_top="middle", pos_left="right")
        )

        # 画图: BadRate
        line_badrate = Line().add_xaxis(list(result["dt"]))
        for b in sub_report["Bin"]:
            this_interval = str(sub_report.loc[b]["Interval"])
            line_badrate.add_yaxis(this_interval, list(result[f"{b}_badrate"]), stack=False)
        line_badrate.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{var}_Bin_BadRate_Trend"),
            datazoom_opts=[opts.DataZoomOpts(is_show=True), ],
            legend_opts=opts.LegendOpts(is_show=True, orient="vertical", pos_top="middle", pos_left="right")
        )
        return {
            "var": var,
            "trend_iv": self.prepare_render(line_iv),
            "trend_prop": self.prepare_render(bar_prop),
            "trend_badrate": self.prepare_render(line_badrate)
        }

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

    def plot_bin(self, filepath):
        info = []
        for var in self.var_list:
            this_res = self.plot_var_trend(self.df, var, self.y, self.time_column)
            info.append(copy(this_res))
        self.render_template("trend_plot.html", filepath, info=info)
