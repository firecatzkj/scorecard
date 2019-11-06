# -*- coding: utf8 -*-
import pandas as pd
import os
import json
import base64
import pyecharts
import xlsxwriter
import pyecharts.options as opts
from copy import copy
from pyecharts.charts import Line, Grid, Page
from ..plot.plot_bin import BinPlot
from ..model import ModelTools
from ..eda import EdaTools
from ..swap import SwapData
from ..benchmark import XGBTrainer
from ..metrics.model import ks_score, auc_score, accum_auc, predict_vs_actual
from jinja2 import Environment, FileSystemLoader
from pyecharts.components import Table
from pyecharts.globals import CurrentConfig
from pyecharts.render.engine import RenderEngine


class ReportData:

    @staticmethod
    def prepare_render(chart):
        """
        pyechart prepare_render
        :param chart:
        :return:
        """
        CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader("{}/template".format(os.path.dirname(__file__))))
        CurrentConfig.ONLINE_HOST = "https://cdn.bootcss.com/echarts/4.2.1-rc1/"
        render_engine = RenderEngine()
        if isinstance(chart, pyecharts.components.Table):
            return chart
        else:
            chart._prepare_render()
            return render_engine.generate_js_link(chart)

    @staticmethod
    def sample_overview(df_train: pd.DataFrame,
                        df_test: pd.DataFrame,
                        df_oot: pd.DataFrame,
                        y: str):
        """
        报告数据_样本总览
        :param df_train:
        :param df_test:
        :param df_oot:
        :param y:
        :param time_column:
        :return:
        """
        result = []
        data_set = {
            "train": df_train,
            "test": df_test,
            "oot": df_oot,
        }
        for sub in data_set.items():
            tmp = {}
            sub_df = sub[1]
            tmp["Data"] = sub[0]
            tmp["Total"] = len(sub_df)
            tmp["Bad"] = len(sub_df[sub_df[y] == 1])
            bad_pct = round((tmp["Bad"] / tmp["Total"]) * 100, 2)
            tmp["BadRate"] = f"{bad_pct}%"
            result.append(copy(tmp))
        result = pd.DataFrame(result)[[
            "Data", "Total", "Bad", "BadRate"
        ]]
        return result

    @staticmethod
    def model_params(dc):
        """
        获取模型的描述信息
        参数估计 标准误 z值 wald卡方 p值 置信下界 置信上界
        :param model_results:
        :return:
        """
        res = ModelTools.ParamEST(dc.model_result_data["model_result"], dc.df_train_woed)
        res = res.round(4)
        return res

    @staticmethod
    def accum_auc_chart(dc):
        """
        累计auc变量的顺序
        :param model_results:
        :return:
        """
        return accum_auc(dc.model_result_data["model_result"], dc.df_train_woed, dc.y, None)

    @staticmethod
    def score_card_data(model_result, all_report: pd.DataFrame):
        """
        输出评分卡
        :param model_result:
        :param all_report:
        :return:
        """
        coef_map = model_result.params.to_dict()
        if "const" in coef_map.keys():
            coef_map.pop("const")
        all_report = all_report[
            all_report["Feature"].isin(list(coef_map.keys()))
        ]
        final_result = []
        for sub in all_report.groupby(by="Feature"):
            sub_df = sub[1][[
                "Feature",
                "Interval",
                "woe"
            ]].copy()
            sub_df["coef"] = coef_map[sub[0]]
            sub_df["score"] = sub_df["woe"] * sub_df["coef"]
            final_result.append(copy(sub_df))
        final_result = pd.concat(final_result, axis=0)
        final_result = final_result.round(4)
        return final_result

    @staticmethod
    def score_card2doc(model_result, all_report: pd.DataFrame, filepath):
        score_card = ReportData.score_card_data(model_result, all_report)
        score_card = pd.DataFrame(score_card)
        score_card.rename({
            "Feature": "字段名",
            "Interval": "分组",
            "woe": "woe",
            "coef": "系数",
            "score": "输入变量取值处理"
        }, inplace=True)

        const = model_result.params.get('const', 0)
        workbook = xlsxwriter.Workbook(f'{filepath}/card_doc.xlsx')
        worksheet = workbook.add_worksheet()

        cell_format = workbook.add_format()
        cell_format.set_bold()
        cell_format.set_font_color('red')

        column_style = workbook.add_format({
            'font_size': 10,  # 字体大小
            'bold': True,  # 是否粗体
            'bg_color': '#101010',  # 表格背景颜色
            'font_color': '#FEFEFE',  # 字体颜色
            'align': 'center',  # 居中对齐
            'top': 1,  # 上边框
            # 后面参数是线条宽度
            'left': 1,  # 左边框
            'right': 1,  # 右边框
            'bottom': 1  # 底边框
        })

        cell_style = workbook.add_format({
            'font_size': 10,  # 字体大小
            'align': 'left',  # 居中对齐
            'top': 1,  # 上边框
            # 后面参数是线条宽度
            'left': 1,  # 左边框
            'right': 1,  # 右边框
            'bottom': 1  # 底边框
        })

        worksheet.write(1, 1, f'model = SUM（各项分值）+截距（{const}）', cell_format)
        worksheet.write(3, 1, 'p=1/(exp(-model)+1)', cell_format)

        col_offset = 7
        row_offset = 1

        col = len(score_card.columns)
        row = len(score_card.index)
        for c in range(len(score_card.columns)):
            worksheet.write(6, c + 1, score_card.columns[c], column_style)

        for i in range(col):
            for j in range(row):
                i_off = i + row_offset
                j_off = j + col_offset
                value = str(score_card.values[j, i])
                worksheet.write(j_off, i_off, value, cell_style)
        workbook.close()

    @staticmethod
    def variable_bin_plot(model_result,
                          df_train: pd.DataFrame,
                          df_test: pd.DataFrame,
                          df_oot: pd.DataFrame):
        """
        这个地方取变量最后确定分箱之后的bin_plot.html, bin_plot.html随all_report一同产出
        :param model_result:
        :param df_train:
        :param df_test:
        :param df_oot:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def ks_auc_data(y_pred_data: dict):
        """
        计算
        :param y_pred_data:
        :return:
        """
        result = []
        for data_set, y_data in y_pred_data.items():
            tmp = {
                "DataSet": data_set,
                "KS": round(ks_score(y_data["y_true"], y_data["y_pred"]), 4),
                "AUC": round(auc_score(y_data["y_true"], y_data["y_pred"]), 4)
            }
            result.append(copy(tmp))
        result = pd.DataFrame(result)[[
            "DataSet",
            "KS",
            "AUC",
        ]]
        return result

    @staticmethod
    def xgb_ks_auc_data(data_x: pd.DataFrame,
                        data_y: pd.Series,
                        data_test: pd.DataFrame,
                        data_oot: pd.DataFrame):
        """
        xgb benchmark模型在三个样本上的ks和auc
        :param data_x:
        :param data_y:
        :param data_test:
        :param data_oot:
        :return:
        """
        xgbt = XGBTrainer(data_x, data_y, grid=True)
        xgbt.fit()
        raise NotImplementedError

    @staticmethod
    def ks_auc_plot(dc, data_dir):
        data_dir = os.path.abspath(data_dir)
        model_version = dc.model_result_data["model_info"]["model_version"]
        model_files = os.listdir(f"{data_dir}/model/v{model_version}")
        print(model_files)
        result = {}
        for mf in model_files:
            if str(mf).startswith("ks_train"):
                result["ks_train"] = f"{data_dir}/model/v{model_version}/{mf}"
            elif str(mf).startswith("ks_test"):
                result["ks_test"] = f"{data_dir}/model/v{model_version}/{mf}"
            elif str(mf).startswith("ks_oot"):
                result["ks_oot"] = f"{data_dir}/model/v{model_version}/{mf}"
            elif str(mf).startswith("roc_train"):
                result["roc_train"] = f"{data_dir}/model/v{model_version}/{mf}"
            elif str(mf).startswith("roc_test"):
                result["roc_test"] = f"{data_dir}/model/v{model_version}/{mf}"
            elif str(mf).startswith("roc_oot"):
                result["roc_oot"] = f"{data_dir}/model/v{model_version}/{mf}"
            else:
                continue

        result_b64 = {}
        for k, v in result.items():
            with open(v, "rb") as f:
                # b64encode是编码，b64decode是解码
                base64_data = base64.b64encode(f.read())
                print(type(base64_data))
                # base64.b64decode(base64data)
                # <img src="data:image/png;base64, {{ ks_auc_plot['ks_train'] }} "/>
                base64_data = str(base64_data, encoding="utf8")
                print(type(base64_data))
                result_b64[k] = f'<img src="data:image/png;base64,{base64_data}"/>'
        return result_b64

    @staticmethod
    def lift_combine(dc):
        result = {}

        lift_combine = Line()
        lift_combine.add_xaxis(list(range(10)))
        for k, v in dc.model_result_data["lift"].items():
            df = v.copy()
            # 图表
            table = Table()
            headers = list(df.columns)
            rows = [list(df.iloc[i]) for i in range(len(df))]
            table.add(headers, rows)

            # 图
            lift_combine.add_yaxis(f"{k}_badrate", list(df["bad_rate"]))
            lift_combine.add_yaxis(f"{k}_pred", list(df["pred"]))
            line = Line(init_opts=opts.InitOpts(width="650px")) \
                .add_xaxis(list(range(len(df)))) \
                .add_yaxis("bad_rate", list(df["bad_rate"]), label_opts=opts.LabelOpts(position="top")) \
                .add_yaxis("pred", list(df["pred"]), label_opts=opts.LabelOpts(position="bottom")) \
                .set_global_opts(title_opts=opts.TitleOpts(title=f"{k}_lift_chart"))

            tmp = {
                "chart": ReportData.prepare_render(line),
                "table": table
            }
            result[k] = copy(tmp)
        lift_combine.set_global_opts(title_opts=opts.TitleOpts(title="Lift on Train|Test|Oot"),
                                     legend_opts=opts.LegendOpts(pos_top=20))
        result["lift_combine"] = ReportData.prepare_render(lift_combine)
        return result

    @staticmethod
    def bin_plot_for_model(dc, report_dir):
        """
        仅仅生成进模型的变量的bin_plot
        :param dc:
        :return:
        """
        var_all = list(dc.model_result_data["model_result"].params.keys())
        if "const" in var_all:
            var_all.remove("const")
        all_report = dc.bin_data["data"]["all_report"]
        change_report = dc.bin_data["data"]["change_report"]

        all_report = all_report[all_report["Feature"].isin(var_all)]
        change_report = change_report[change_report["Feature"].isin(var_all)]
        bp = BinPlot(
            df=dc.train,
            all_report=all_report,
            change_report=change_report,
            y=dc.y,
            df_test=dc.test,
            df_oot=dc.oot
        )
        bp.plot_bin(all_report, f"{report_dir}/bin_plot.html")

    @staticmethod
    def pva_data(dc):
        result = []
        var_all = list(dc.model_result_data["model_result"].params.keys())
        if "const" in var_all:
            var_all.remove("const")

        change_report = dc.bin_data["data"]["change_report"]
        change_report = change_report[change_report["Feature"].isin(var_all)]

        df_train = dc.train
        y_true_train = dc.model_result_data["y_pred_data"]["train"]["y_true"]
        y_pred_train = dc.model_result_data["y_pred_data"]["train"]["y_pred"]

        df_test = dc.test
        y_true_test = dc.model_result_data["y_pred_data"]["test"]["y_true"]
        y_pred_test = dc.model_result_data["y_pred_data"]["test"]["y_pred"]

        df_oot = dc.oot
        y_true_oot = dc.model_result_data["y_pred_data"]["oot"]["y_true"]
        y_pred_oot = dc.model_result_data["y_pred_data"]["oot"]["y_pred"]

        for i in range(len(change_report)):
            sub = change_report.iloc[i]
            if sub["type"] == "C":
                this_dtype = "num"
            elif sub["type"] == "M":
                this_dtype = "char"
            else:
                this_dtype = "char"
            pva_train = predict_vs_actual(df_train[sub["Feature"]],
                                    this_dtype,
                                    y_true_train,
                                    y_pred_train,
                                    json.loads(sub["Interval"]))
            pva_train.set_global_opts(title_opts=opts.TitleOpts(title="train"))

            pva_test = predict_vs_actual(df_test[sub["Feature"]],
                                          this_dtype,
                                          y_true_test,
                                          y_pred_test,
                                          json.loads(sub["Interval"]))
            pva_test.set_global_opts(title_opts=opts.TitleOpts(title="test"))

            pva_oot = predict_vs_actual(df_oot[sub["Feature"]],
                                         this_dtype,
                                         y_true_oot,
                                         y_pred_oot,
                                         json.loads(sub["Interval"]))
            pva_oot.set_global_opts(title_opts=opts.TitleOpts(title="oot"))


            pva = Page(layout=Page.SimplePageLayout)
            pva.add(pva_train, pva_test, pva_oot)

            tmp = {
                "var": sub["Feature"],
                "chart": ReportData.prepare_render(pva)
            }
            result.append(copy(tmp))
        return result

    @staticmethod
    def eda_plot(dc):
        var_all = list(dc.model_result_data["model_result"].params.keys())
        if "const" in var_all:
            var_all.remove("const")
        res = []
        for var in var_all:
            this_chart = EdaTools.plot_fluctuation_by_time(dc.train, var, dc.y, dc.time_column)
            tmp = {
                "var": var,
                "chart": copy(this_chart)
            }
            res.append(copy(tmp))
        return res

    @staticmethod
    def swap_result(**kwargs):
        swap_res = {}
        P1 = kwargs["P1"]
        P2 = kwargs["P2"]
        P1_rank, P2_rank, df_swap_len, df_swap_sum, df_swap_len_rate, df_swap_sum_rate = SwapData(**kwargs)
        # 评分排序
        P1_rank[P1] = P1_rank.apply(lambda x: round(x["sum"] / x["len"], 4), axis=1)
        P2_rank[P2] = P2_rank.apply(lambda x: round(x["sum"] / x["len"], 4), axis=1)

        score_rank = pd.concat([
            P1_rank[[P1, ]],
            P2_rank[[P2, ]],
        ], axis=1)
        score_rank.columns = [f"{P1}_rank", f"{P2}_rank"]
        swap_res["score_rank"] = score_rank
        # 逾期率
        swap_res["swap_badrate"] = df_swap_sum_rate.round(4)
        # 人数占比
        swap_res["swap_prop"] = df_swap_len_rate.round(4)
        # 逾期人数分布
        swap_res["swap_bad_num"] = df_swap_sum
        # 人数分布
        swap_res["swap_num"] = df_swap_len

        return swap_res
