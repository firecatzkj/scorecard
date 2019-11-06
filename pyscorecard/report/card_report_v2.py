# -*- coding: utf8 -*-
# -*- coding: utf8 -*-
"""
生成模型报告
"""
import os
import shutil
import pyecharts
import pandas as pd
from copy import copy
from .data_handler import ReportData
from ..metrics.model import corr_heat_map
from ..tools import heat_map_common
from jinja2 import Environment, FileSystemLoader
from pyecharts.components import Table
from pyecharts.globals import CurrentConfig
from pyecharts.render.engine import RenderEngine
from pyecharts.commons.utils import write_utf8_html_file


class Report:
    def __init__(self, dc, report_dir: str, data_dir: str, swap_params, **info):
        """
        生成报告
        :param dc:
        :param report_dir:
        """
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader("{}/template".format(os.path.dirname(__file__))))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()
        self.dc = dc
        self.report_dir = report_dir
        self.data_dir = data_dir
        self.swap_params = swap_params
        self.info = info

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

    def df2table(self, df: pd.DataFrame):
        table = Table()
        headers = list(df.columns)
        rows = [list(df.iloc[i]) for i in range(len(df))]
        table.add(headers, rows)
        return table

    def prepare_report(self):
        # 样本概述
        overview_data_df = ReportData.sample_overview(self.dc.train, self.dc.test, self.dc.oot, self.dc.y)
        overview_data = self.df2table(overview_data_df)

        # ks auc
        ks_auc_data_df = ReportData.ks_auc_data(self.dc.y_pred_data)
        ks_auc_data = self.df2table(ks_auc_data_df)
        ks_plot_b64 = ReportData.ks_auc_plot(self.dc, self.data_dir)

        # lift chart
        lift_ks_data = ReportData.lift_combine(self.dc)

        # 累计auc
        accum_auc_chart = self.prepare_render(ReportData.accum_auc_chart(self.dc))

        # 评分卡
        scorecard_data_df = ReportData.score_card_data(self.dc.model_result_data["model_result"],
                                                    self.dc.bin_data["data"]["all_report"])
        scorecard_data = self.df2table(scorecard_data_df)

        # 变量相关性
        cols = list(self.dc.model_result.params.keys())
        if "const" in cols:
            cols.remove("const")
        corr = corr_heat_map(self.dc.df_train_woed, cols)
        corr = self.prepare_render(corr)

        # 模型总结
        model_summary_df = ReportData.model_params(self.dc)
        model_summary = self.df2table(model_summary_df)

        # 变量predict vs actual
        pva_res = ReportData.pva_data(self.dc)

        # 变量eda情况
        eda_res = ReportData.eda_plot(self.dc)

        # swap
        if self.swap_params is not None:
            swap_res_new = ReportData.swap_result(**self.swap_params)
            # swap_res = {k: self.df2table(v) for k, v in swap_res_new.items()}
            swap_res = {}
            for k, v in swap_res_new.items():
                print(k, "??????????????")
                if k == "score_rank":
                    swap_res[k] = self.df2table(v)
                else:
                    tmp = self.prepare_render(heat_map_common(v, k))
                    swap_res[k] = copy(tmp)
        else:
            swap_res = None

        res_final = {
            "overview_data": overview_data,
            "ks_auc_data": ks_auc_data,
            "ks_auc_plot": ks_plot_b64,
            "accum_auc_chart": accum_auc_chart,
            "lift": lift_ks_data,
            "scorecard_data": scorecard_data,
            "corr": corr,
            "model_summary": model_summary,
            "pva_res": pva_res,
            "eda_res": eda_res,
            "swap_res": swap_res
        }

        return res_final

    def generate_report(self, data_dir: str):
        """
        生成报告
        :param report_dir:
        :return:
        """
        report_dir = f"{data_dir}/report/"
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)

        res_final = self.prepare_report()
        if self.swap_params is None:
            self.render_template("model_desc_ns.html",
                                 f"{report_dir}/report_model_desc.html",
                                 **res_final)
        else:
            self.render_template("model_desc.html",
                                 f"{report_dir}/report_model_desc.html",
                                 **res_final)

        eda_dir = f"{report_dir}/eda/"
        os.makedirs(eda_dir)
        eda_res = res_final["eda_res"]
        for sub in eda_res:
            var = sub["var"]
            chart = sub["chart"]
            chart.render(f"{eda_dir}/{var}.html")
        ReportData.bin_plot_for_model(self.dc, report_dir)
        ReportData.score_card2doc(self.dc.model_result_data["model_result"],
                                  self.dc.bin_data["data"]["all_report"],
                                  report_dir)


