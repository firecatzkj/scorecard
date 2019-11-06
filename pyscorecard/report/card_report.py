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
# from ..metrics.model import accum_auc, roc_plot_html
from jinja2 import Environment, FileSystemLoader
from pyecharts.components import Table
from pyecharts.charts import Page, Line
from pyecharts.globals import CurrentConfig
from pyecharts.render.engine import RenderEngine
from pyecharts.commons.utils import write_utf8_html_file


class Report:
    def __init__(self,
                 report_dir: str,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 df_oot: pd.DataFrame,
                 df_train_woed: pd.DataFrame,
                 df_test_woed: pd.DataFrame,
                 df_oot_woed: pd.DataFrame,
                 all_report: pd.DataFrame,
                 y: str,
                 model_results,
                 bin_plot_html_path: str
                 ):
        """
        参数有点多,但会封装起来
        :param report_dir:
        :param df_train:
        :param df_test:
        :param df_oot:
        :param df_train_woed:
        :param df_test_woed:
        :param df_oot_woed:
        :param all_report:
        :param y:
        :param model_results:
        :param bin_plot_html_path:
        """
        self.df_train = df_train
        self.df_test = df_test
        self.df_oot = df_oot
        self.df_train_woed = df_train_woed
        self.df_test_woed = df_test_woed
        self.df_oot_woed = df_oot_woed
        self.all_report = all_report
        self.y = y
        self.model_results = model_results
        self.bin_plot_html_path = bin_plot_html_path
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader("{}/template".format(os.path.dirname(__file__))))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        self.render_engine = RenderEngine()
        self.report_dir = report_dir

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

    def page_sample_overview(self):
        overview_data = ReportData.sample_overview(
            self.df_train,
            self.df_test,
            self.df_oot,
            self.y
        )
        table = Table()
        headers = list(overview_data.columns)
        rows = [list(overview_data.iloc[i]) for i in range(len(overview_data))]
        table.add(headers, rows)
        return self.prepare_render(table)

    def page_model_params(self):
        param_data = ReportData.model_params(self.model_results)
        table = Table()
        headers = list(param_data.columns)
        rows = [list(param_data.iloc[i]) for i in range(len(param_data))]
        table.add(headers, rows)
        return self.prepare_render(table)

    def page_lift_ks_all(self, page_path):
        lk_data, y_data = ReportData.lift_ks_all(
            self.model_results,
            self.df_train_woed,
            self.df_test_woed,
            self.df_oot_woed,
            self.y
        )
        #print(lk_data)
        #print(y_data)
        # 渲染table数据
        table_map = {}
        for sub in lk_data.items():
            table = Table()
            headers = list(sub[1].columns)
            rows = [list(sub[1].iloc[i]) for i in range(len(sub[1]))]
            table.add(headers, rows)
            table = self.prepare_render(table)
            table_map[sub[0]] = copy(table)
        self.render_template("page_lift_ks.html", f"{page_path}/lift_ks.html", **table_map)
        # 渲染图表数据
        # roc_map = {}
        # for sub in y_data.items():
        #     roc_map[sub[0]] = self.prepare_render(
        #         roc_plot_html(sub[1]["y_true"], sub[1]["y_pred"])
        #     )
        # self.render_template("roc_all.html", f"{page_path}/roc.html", **roc_map)

        roc_map = {}
        for sub in lk_data.items():
            roc_map[sub[0]] = self.prepare_render(
                copy(Line().add_xaxis(list(range(len(sub[1]))))
                    .add_yaxis("bad_rate", list(sub[1]["bad_rate"]))
                    .add_yaxis("KS", list(sub[1]["ks_score"])))
            )
        print(roc_map.keys())
        self.render_template("roc_all.html", f"{page_path}/roc.html", **roc_map)

        # roc_map = {}
        # l1 = Line().add_xaxis(list(range(len(lk_data["train"]))))\
        #             .add_yaxis("bad_rate", list(lk_data["train"]["bad_rate"]))\
        #             .add_yaxis("KS", list(lk_data["train"]["ks_score"]))
        #
        # l2 = Line().add_xaxis(list(range(len(lk_data["test"])))) \
        #     .add_yaxis("bad_rate", list(lk_data["test"]["bad_rate"])) \
        #     .add_yaxis("KS", list(lk_data["test"]["ks_score"]))
        #
        # l3 = Line().add_xaxis(list(range(len(lk_data["oot"])))) \
        #     .add_yaxis("bad_rate", list(lk_data["oot"]["bad_rate"])) \
        #     .add_yaxis("KS", list(lk_data["oot"]["ks_score"]))
        # roc_map["train"] = l1
        # roc_map["test"] = l2
        # roc_map["oot"] = l3
        # self.render_template("roc_all.html", f"{page_path}/roc.html", **roc_map)





    # def page_accum_auc(self):
    #     var_list = ReportData.accum_auc_var(self.model_results)
    #     chart = accum_auc(var_list, self.df_train_woed, self.y, None)
    #     return self.prepare_render(chart)

    def page_scorecard(self):
        """
        渲染评分卡的页面
        :return:
        """
        score_card_data = ReportData.score_card_data(
            self.model_results,
            self.all_report
        )

        table = Table()
        headers = list(score_card_data.columns)
        rows = [list(score_card_data.iloc[i]) for i in range(len(score_card_data))]
        table.add(headers, rows)
        return self.prepare_render(table)

    def generate_report(self):
        """
        生成报告
        :param report_dir:
        :return:
        """
        if os.path.exists("{}/report".format(self.report_dir)):
            shutil.rmtree("{}/report".format(self.report_dir))
        os.mkdir("{}/report".format(self.report_dir))
        os.mkdir("{}/report/page".format(self.report_dir))
        shutil.copy("{}/template/report.html".format(os.path.dirname(__file__)),
                    "{}/report".format(self.report_dir))
        page_path = "{}/report/page/".format(self.report_dir)
        # do: 后面需要实现每个页面生成的代码
        # 1. 数据总览
        self.page_sample_overview().render(f"{page_path}/sample_overview.html")
        # 2. 模型系数
        self.page_model_params().render(f"{page_path}/model_params.html")
        # 3. 评分卡
        self.page_scorecard().render(f"{page_path}/scorecard.html")
        # 4. lift ks
        self.page_lift_ks_all(page_path)
        # 5. 累积auc
        self.page_accum_auc().render(f"{page_path}/accum_auc.html")
        # 6. 变量分bin三个样本对比
        shutil.copy(self.bin_plot_html_path, f"{page_path}/bin_plot.html")
        print("Enjoy! ^_^ ")



