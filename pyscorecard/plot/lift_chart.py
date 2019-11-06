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

# print(os.path.abspath(os.path.dirname(__file__)) + "/template")
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 2500)


class LiftChart:

    def __init__(self, df_lift: pd.DataFrame):
        template_path = os.path.abspath(os.path.dirname(__file__)) + "/template"
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader(template_path))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()
        self.df = df_lift

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
        # c_map = {}
        # for i in kwargs.items():
        #     c_map[i[0]] = self.prepare_render(i[1])
        html = tpl.render(**kwargs)
        write_utf8_html_file(filename, self.render_engine._replace_html(html))

    def plot(self, save_label, out_dir):
        # 图表
        table = Table()
        headers = list(self.df.columns)
        rows = [list(self.df.iloc[i]) for i in range(len(self.df))]
        table.add(headers, rows)

        # 图
        line = Line() \
            .add_xaxis(list(range(len(self.df)))) \
            .add_yaxis("bad_rate", list(self.df["bad_rate"])) \
            .set_global_opts(title_opts=opts.TitleOpts(title=save_label))
        line = self.prepare_render(line)

        file_path = f"{out_dir}/{save_label}.html"
        self.render_template("lift_chart.html", file_path, chart=line, table=table)


class LiftChartCombine:

    def __init__(self, lift_train: pd.DataFrame, lift_test: pd.DataFrame, lift_oot: pd.DataFrame):
        """
        把train, test, oot三个数据集的lift chart放在一起来看
        :param lift_train:
        :param lift_test:
        :param lift_oot:
        """
        template_path = os.path.abspath(os.path.dirname(__file__)) + "/template"
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader(template_path))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()
        self.lift_train = lift_train
        self.lift_test = lift_test
        self.lift_oot = lift_oot

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

    def plot(self, filepath):
        pass
