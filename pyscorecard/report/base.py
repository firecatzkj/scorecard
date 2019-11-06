# -*- coding: utf8 -*-
import os
import shutil
import pyecharts
import pandas as pd
from copy import copy
from .data_handler import ReportData
# from ..metrics.model import accum_auc, roc_plot_html
from jinja2 import Environment, FileSystemLoader
from pyecharts.components import Table
from pyecharts.charts import Page, Line
from pyecharts.globals import CurrentConfig
from pyecharts.render.engine import RenderEngine
from pyecharts.commons.utils import write_utf8_html_file


class MyRender:
    def __init__(self):
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader("{}/template".format(os.path.dirname(__file__))))
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
