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


class BinDiffY:

    def __init__(self, df: pd.DataFrame, var_list: list, y_list: list, change_report: pd.DataFrame):
        template_path = os.path.abspath(os.path.dirname(__file__)) + "/template"
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader(template_path))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()
        self.df = df
        self.var_list = var_list
        self.y_list = y_list
        self.change_report = change_report.set_index("Feature")

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

    def plot_iv_badrate(self, var: str):
        """
        绘制变量再不同的y上面的iv变化
        :param var:
        :return:
        """

        # 计算all_report
        all_reports = {}
        for y in self.y_list:
            this_df = self.df[[var, y]]
            cuts = json.loads(self.change_report.loc[var]["Interval"])
            ftype = self.change_report.loc[var]["type"]
            sub_report = self.generate_sub_all_report(this_df, var, y, cuts, ftype)
            sub_report = sub_report.round(4)
            all_reports[y] = sub_report

        # 准备iv的画图数据
        iv_data = []
        for y, sub_report in all_reports.items():
            tmp = {
                "y": y,
                "iv": sub_report["iv"][0]
            }
            iv_data.append(copy(tmp))
        iv_data = pd.DataFrame(iv_data)

        # 准备badrate的画图数据
        badrate_data = []
        for y, sub_report in all_reports.items():
            tmp = {
                "Interval": sub_report["Interval"].astype(str),
                y: sub_report["PD"],
            }
            tmp = pd.DataFrame(tmp)
            tmp = tmp.set_index("Interval")
            badrate_data.append(tmp)
        badrate_all = pd.concat(badrate_data, axis=1)

        # 画图
        iv_bar = Bar()\
            .add_xaxis(list(iv_data["y"]))\
            .add_yaxis("IV", list(iv_data["iv"]))

        bd_line = Line()
        bd_line.add_xaxis(list(badrate_all.index))
        for y in self.y_list:
            bd_line.add_yaxis(y, list(badrate_all[y]))
        bd_line.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)))
        return self.prepare_render(iv_bar), self.prepare_render(bd_line)

    def plot(self, filepath):
        info = []
        for v in self.var_list:
            res = self.plot_iv_badrate(v)
            tmp = {
                "var": v,
                "iv": res[0],
                "pd": res[1]
            }
            info.append(copy(tmp))
        self.render_template("diff_y.html", filepath, info=info)
