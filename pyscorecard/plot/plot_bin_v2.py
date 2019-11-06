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
    doing todo
    """
    def __init__(self,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 df_oot: pd.DataFrame,
                 y: str,
                 all_report_train,
                 change_report_train):
        """
        v2
        :param df_train:
        :param df_test:
        :param df_oot:
        :param y:
        :param all_report_train:
        :param change_report_train:
        """

        self.df_train = df_train
        self.df_test = df_test
        self.df_oot = df_oot
        self.y = y
        self.all_report_train = all_report_train
        self.change_report_train = change_report_train
        template_path = os.path.abspath(os.path.dirname(__file__)) + "/template"
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader(template_path))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()

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


