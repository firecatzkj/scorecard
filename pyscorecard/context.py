# -*- coding: utf8 -*-
import os
import shutil
import pandas as pd
import numpy as np
import pandas_profiling as pdf
from copy import copy
from .eda import eda_report_html
from .features import FeatureSelect
from .Tool import Bins
from .plot.plot_bin import BinPlot
from .plot.plot_bin_trend import BinTrendPlot
from .plot.plot_diff_y import BinDiffY
from .tools import df2csv
from .model import ModelTools, mylogit, model_predict
from .eda import EdaTools
from .metrics.model import lift_ks, roc_plot, accum_auc, ks_plot
from .metrics.model import corr_heat_map
from .report.card_report_v2 import Report
from .tools import save_to_pkl, interval_round, load_from_pkl, dict2file


class DataContext:

    def __init__(self,
                 y: str,
                 train: pd.DataFrame,
                 test: pd.DataFrame,
                 oot: pd.DataFrame,
                 dtypes: dict,
                 uid: str=None,
                 time_column: str=None,
                 data_dir: str=None):
        """
        :param train: 训练集
        :param test: 测试集
        :param oot: 时间外样本如果没有填None
        :param dtypes:
        """
        self.y = y
        self.x_selected_record = self.x_record_init(list(train.drop(y, axis=1).columns))
        self.dtypes = dtypes
        self.uid = uid
        self.train = self.data_schema(train)
        self.test = self.data_schema(test)
        self.oot = self.data_schema(oot)
        self.time_column = time_column
        self.bin_version = 0
        self.model_version = 0
        self.eda_filtered = []
        self.feature_selected = []
        self.data_dir = data_dir
        self.bin_data = None
        self.model_result_data = None

    def save_all(self, out_dir):
        filepath = f"{out_dir}/context.pkl"
        save_to_pkl(self, filepath)

    def x_record_init(self, x_list):
        df_init = pd.DataFrame({
            "x": x_list,
            "is_selected": True,
            "drop_reason": None
        })
        return df_init.set_index("x")

    def set_x_selected(self, x_list: list):
        """
        手动直接设置x_selected
        :param x_list:
        :return:
        """
        self.x_selected_record["is_selected"] = True
        self.x_selected_record["drop_reason"] = None
        x_all = self.x_selected_record.index
        for i in x_all:
            if i not in x_list:
                self.drop_x("manual", i)

    @property
    def x_selected(self):
        res = self.x_selected_record[self.x_selected_record["is_selected"]].index
        return list(res)

    @property
    def all_report(self):
        if self.bin_data is None:
            return None
        else:
            return self.bin_data["data"]["all_report"]

    @property
    def change_report(self):
        if self.bin_data is None:
            return None
        else:
            return self.bin_data["data"]["change_report"]

    @property
    def df_train_woed(self):
        if self.bin_data is None:
            return None
        else:
            return self.bin_data["data"]["df_train_woed"]

    @property
    def df_test_woed(self):
        if self.bin_data is None:
            return None
        else:
            return self.bin_data["data"]["df_test_woed"]

    @property
    def df_oot_woed(self):
        if self.bin_data is None:
            return None
        else:
            return self.bin_data["data"]["df_oot_woed"]

    @property
    def current_bin_data_version(self):
        if self.bin_data is None:
            return None
        else:
            return self.bin_data["version"]

    @property
    def model_info(self):
        if self.model_result_data is None:
            return None
        else:
            return self.model_result_data["model_info"]

    @property
    def model_result(self):
        if self.model_result_data is None:
            return None
        else:
            return self.model_result_data["model_result"]

    @property
    def y_pred_data(self):
        if self.model_result_data is None:
            return None
        else:
            return self.model_result_data["y_pred_data"]

    def drop_x(self, reason, *x_list):
        """
        删除x
        :param reason: 理由
        :param x_list: x列表
        :return:
        """
        for i in x_list:
            self.x_selected_record.loc[i, "is_selected"] = False
            self.x_selected_record.loc[i, "drop_reason"] = reason
        return x_list

    def add_x(self, *x_list):
        """
        添加x
        :param x_list:
        :return:
        """
        for i in x_list:
            self.x_selected_record.loc[i, "is_selected"] = True
            self.x_selected_record.loc[i, "drop_reason"] = None
        print(f"variables: {x_list} added!")

    def save_object(self, name, obj):
        """
        方便建模过程中存储相关的context数据集
        :param name: key
        :param obj: object
        :return:
        """
        self.__setattr__(name, obj)

    def get_object(self, name):
        """
        获取内部属性
        :param name:
        :return:
        """
        return self.__getattribute__(name)

    def data_schema(self, df: pd.DataFrame):
        """
        对数据集进行数据类型的矫正
        :param df:
        :return:
        """
        for k in self.dtypes.keys():
            if k not in df.columns:
                print("'{}' not in your dataset, please check your params and dataset".format(k))
            try:
                if self.dtypes[k] == "M":
                    df[k] = df[k].astype("str")
                elif self.dtypes[k] == "C":
                    df[k] = df[k].astype("float")
                else:
                    df[k] = df[k].astype(self.dtypes[k])
            except ValueError as e:
                continue
        return df

    def _eda_report_saver(self, report: dict, report_prefix: str, report_dir: str):
        """
        把pandas_profiling的结果集分类存储
        :param report:
        :param report_prefix: 区分的前缀
        :param report_dir: 报告存储路径
        :return:
        """
        location_prefix = report_dir + "/" + report_prefix + "_{}.csv"
        correlations = report["correlations"]
        corr_pearson = correlations["pearson"]
        corr_spearman = correlations["spearman"]
        variables = report["variables"]
        del variables["histogram"]
        del variables["mini_histogram"]
        variables["varname"] = variables.index
        columns_all = list(variables.columns)
        columns_all.remove("varname")
        columns_all.insert(0, "varname")
        variables = variables[columns_all]
        corr_pearson.to_csv(location_prefix.format("corr_pearson"), index=True, encoding="utf8")
        corr_spearman.to_csv(location_prefix.format("corr_spearman"), index=True, encoding="utf8")
        variables.to_csv(location_prefix.format("variables"), index=False, encoding="utf8")

    def eda_report_all(self):
        """
        对train, test, oot
        :param report_dir:
        :return:
        """
        report_dir = self.data_dir + "/eda/"
        if os.path.exists(report_dir):
            pass
        else:
            os.makedirs(report_dir)
        print("Start to generate EDA report! ")
        eda_train = eda_report_html(self.train[self.x_selected],
                                    {},
                                    report_dir + "eda_train.html")
        eda_test = eda_report_html(self.test[self.x_selected],
                                   {},
                                   report_dir + "eda_test.html")
        eda_oot = eda_report_html(self.oot[self.x_selected],
                                  {},
                                  report_dir + "eda_oot.html")

        eda_train_desc = eda_train.get_description()
        eda_test_desc = eda_test.get_description()
        eda_oot_desc = eda_oot.get_description()
        # 合并train test oot
        data_list = {
            "train": eda_train_desc["variables"],
            "test": eda_test_desc["variables"],
            "oot": eda_oot_desc["variables"]
        }
        var_report_all = []
        for sub in data_list.items():
            tmp = sub[1].copy()
            tmp["data_flag"] = sub[0]
            del tmp["histogram"]
            del tmp["mini_histogram"]
            tmp["varname"] = tmp.index
            columns_all = list(tmp.columns)
            columns_all.remove("varname")
            columns_all.insert(0, "varname")
            columns_all.remove("data_flag")
            columns_all.insert(1, "data_flag")
            tmp = tmp[columns_all]
            var_report_all.append(copy(tmp))
        var_report_all = pd.concat(var_report_all, axis=0).sort_values(by="varname")
        var_report_all.to_csv(f"{report_dir}/eda_variable_all.csv", encoding="utf8")
        # 存储eda的数据到report_dir ==> *.csv
        self._eda_report_saver(eda_train_desc, "eda_train", report_dir)
        self._eda_report_saver(eda_test_desc, "eda_test", report_dir)
        self._eda_report_saver(eda_oot_desc, "eda_oot", report_dir)
        print("EDA report finished, you can find report in {}".format(report_dir))
        self.__setattr__("eda_train", eda_train)
        self.__setattr__("eda_test", eda_test)
        self.__setattr__("eda_oot", eda_oot)
        print("Context updated you can get EDA report dataframe by context.get_object('eda_train')")

    def eda_filter(self,
                   dataset="train",
                   reject=None,
                   corr=None,
                   missing=None,
                   spar=None,
                   uniq=None,
                   adjust_x=True):
        """
        通过eda报告筛变量
        :param dataset:
        :param reject: 相关性, 两个相关性高的变量,删除缺失高的那个
        :param corr: 按照相关性的阈值来删除变量,建议0.9以上
        :param missing: 缺失率, 删除高缺失变量的方法
        :param spar: 稀疏比例, 稀疏比例高的变量
        :param uniq: 对应的类别型而且基本是一对一的变量：身份证类，号码类
        :param adjust_x:
        :return: 删除的变量
        """
        var_filtered = {}
        try:
            report = self.get_object(f"eda_{dataset}")
        except AttributeError as e:
            report = pdf.ProfileReport(self.get_object(dataset))
        if reject:
            var_filter_reject = EdaTools.profile_filter_corr(report, reject)
            var_filtered.update(dict.fromkeys(var_filter_reject, "reject"))
        if corr:
            cur_data = self.get_object(dataset)
            var_filter_corr = EdaTools.corr_filter(cur_data, corr)
            var_filtered.update(dict.fromkeys(var_filter_corr, "corr"))

        if missing:
            var_filter_miss = EdaTools.profile_filter_missing(report, missing)
            var_filtered.update(dict.fromkeys(var_filter_miss, "missing"))
        if spar:
            var_filter_spar = EdaTools.profile_filter_spar(report, spar)
            var_filtered.update(dict.fromkeys(var_filter_spar, "spar"))
        if uniq:
            var_filter_uniq = EdaTools.profile_filter_uniq(report, uniq)
            var_filtered.update(dict.fromkeys(var_filter_uniq, "uniq"))
        self.save_object("eda_filtered", var_filtered)
        print("eda_filtered已更新!")
        # 是否根据筛选结果剔除变量
        if adjust_x:
            for k, v in var_filtered.items():
                self.drop_x(v, k)
            print("dc.x_selected 已经删除eda的过滤结果")

    def feature_report(self, x: list=None, data_set="train"):
        """
        单变量筛选报告
        :param x: 默认None, 如果指定,就会用x给定的变量
        :param report_dir:
        :param data_set:
            - 三种类型: train,test,oot
            - 默认是train的报告
        :return:
        """
        report_dir = self.data_dir + "/feature_select/"
        if os.path.exists(report_dir):
            pass
        else:
            os.makedirs(report_dir)

        report_name = "{}.csv".format(data_set)
        print("Feature selection started take a cup of coffee!")
        if x:
            x.append(self.y)
            data = self.get_object(data_set)[x]
        else:
            x = self.x_selected
            x.append(self.y)
            data = self.get_object(data_set)[x]
        fs = FeatureSelect(data, self.y, {}, do_binning=True, df_test=self.test)
        report = fs.get_combine_result(grid_search=False, param=None)
        if report_dir:
            report.to_csv(report_dir + report_name, index=False, encoding="utf8")
            print("Feture report saved in out/{}".format(report_name))
        # 存档
        self.save_object(f"feature_report_{data_set}", report)
        print("Feature selection finished have fun! :)")

    def feature_report_all(self):
        """
        单变量筛选报告, train test oot 合并结果集
        :param report_dir:
        :return:
        """
        print("Feature select report start")
        report_dir = self.data_dir + "/feature_select/"
        if os.path.exists(report_dir):
            pass
        else:
            os.makedirs(report_dir)

        col_selected = self.x_selected
        col_selected.append(self.y)
        data_list = {
            "train": self.train[col_selected],
            "test": self.test[col_selected],
            "oot": self.oot[col_selected]
        }
        fs_report = []
        for sub in data_list.items():
            if sub[0] == "train":
                fs = FeatureSelect(sub[1], self.y, {}, do_binning=True, df_test=self.oot)
            elif sub[0] == "test":
                fs = FeatureSelect(sub[1], self.y, {}, do_binning=True, df_test=self.oot)
            else:
                fs = FeatureSelect(sub[1], self.y, {}, do_binning=True, df_test=None)
            report = fs.get_combine_result(grid_search=False, param=None)
            report = pd.DataFrame(report)
            report = report.add_prefix("{}_".format(sub[0]))
            report["varname"] = report["{}_{}".format(sub[0], "var_code")]
            del report["{}_{}".format(sub[0], "var_code")]
            fs_report.append(copy(report))
        res = pd.merge(
            fs_report[0],
            fs_report[1],
            how="inner",
            on="varname"
        )
        res = pd.merge(
            res,
            fs_report[2],
            how="inner",
            on="varname"
        )
        columns_all = list(res.columns)
        columns_all.sort()
        res = res[columns_all]
        self.save_object("feature_report_df", res)
        res.index = res["varname"]
        res.to_csv(f"{report_dir}/feature_all.csv", index=True, encoding="utf8")

    def binning_init(self, X: list=None, method=3, end_num=5, mono=False, threshold_value=np.nan, MDP=True, jn=0, bin_num=20):
        """
        调用Bins.generate_raw做分箱的初始化
        @ JiangGuixiang
        Gini: method=2, threshold: 0.0001
        卡方: method=3, threshold: 0.0001*N/end_num

        :param: method: 分箱方法
        :param: end_num: 默认最多分5个箱
        :return:
        """
        # 覆盖当前version, version可以手动设置
        # self.bin_version = self.bin_version + 1
        report_dir = f"{self.data_dir}/bins/v{self.bin_version}/"
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)
        if X:
            cur_selected = X
        else:
            cur_selected = self.x_selected
        if (method == 2) and (threshold_value is np.nan):
            threshold_value = 0.0001
        if (method == 3) and (threshold_value is np.nan):
            threshold_value = (0.0001 * len(self.train)) / end_num
        bins = Bins(method=method, end_num=end_num, threshold_value=threshold_value)
        cur_df_x = self.train[cur_selected]
        cur_df_y = self.train[self.y]
        if mono:
            bin_res = bins.generate_raw_mono(cur_df_x, cur_df_y, type_dict=self.dtypes, jn=jn)
            all_report, change_report, woe_df, bin_df, false_dict = bin_res
        else:
            bin_res = bins.generate_raw(cur_df_x, cur_df_y, type_dict=self.dtypes, MDP=MDP)
            all_report, change_report, woe_df, bin_df, false_dict = bin_res

        # 生成woe df
        df_train_woed = woe_df.copy()
        df_train_woed[self.y] = self.train[self.y]
        df_test_woed = bins.whole_woe_replace(self.test, all_report)
        df_test_woed[self.y] = self.test[self.y]
        df_oot_woed = bins.whole_woe_replace(self.oot, all_report)
        df_oot_woed[self.y] = self.oot[self.y]

        # 生成bin_plot.html
        all_report["interval"] = all_report["Interval"].apply(interval_round)
        all_report["Interval"] = all_report["Interval"].apply(lambda x: str(x))
        bp = BinPlot(
            df=self.train,
            all_report=all_report,
            change_report=change_report,
            y=self.y,
            df_test=self.test,
            df_oot=self.oot,
            bin_num=bin_num
        )
        bp.plot_bin(all_report, f"{report_dir}/bin_plot_{self.bin_version}.html")

        # 生成长期趋势图
        if self.time_column:
            btp = BinTrendPlot(self.train, cur_selected, self.y, self.time_column, change_report)
            btp.plot_bin(f"{report_dir}/bin_trend_plot_{self.bin_version}.html")

        # 存文件
        df2csv(all_report, report_dir, f"all_report_{self.bin_version}.csv")
        df2csv(change_report, report_dir, f"change_report_{self.bin_version}.csv")
        df2csv(df_train_woed, report_dir, "df_train_woed.csv")
        df2csv(df_test_woed, report_dir, "df_test_woed.csv")
        df2csv(df_oot_woed, report_dir, "df_oot_woed.csv")

        # 存pkl
        data_all = {
            "version": self.bin_version,
            "data": {
                "all_report": all_report,
                "change_report": change_report,
                "df_train_woed": df_train_woed,
                "df_test_woed": df_test_woed,
                "df_oot_woed": df_oot_woed
            }
        }
        save_to_pkl(data_all, f"{report_dir}/all.pkl")

        # 存更新context
        # self.save_object("all_report", all_report)
        # self.save_object("change_report", change_report)
        # self.save_object("df_train_woed", df_train_woed)
        # self.save_object("df_test_woed", df_test_woed)
        # self.save_object("df_oot_woed", df_oot_woed)
        self.save_object("bin_data", data_all)
        self.save_object("bin_plot_html", f"{report_dir}/bin_plot_{self.bin_version}.html")

    def binning_from_change(self,
                            change_report: pd.DataFrame,
                            adjust_x: bool=True,
                            bin_num=20):
        """
        ﻿修改change_report,重新生成相关文件并存档
        :param report_dir:
        :param change_report:
        :param adjust_x: 是否根据change_report来调整context的x_selected
        :param kwargs:
        :return:
        """
        # 覆盖当前version,version可以手动设置
        # self.bin_version = self.bin_version + 1
        report_dir = f"{self.data_dir}/bins/v{self.bin_version}/"
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)
        os.mkdir(report_dir)
        bins = Bins()
        all_report_new = bins.mannual_rebin(self.train, change_report, self.train[self.y])
        df_train_woed = bins.whole_woe_replace(self.train, all_report_new)
        df_train_woed[self.y] = self.train[self.y]
        df_test_woed = bins.whole_woe_replace(self.test, all_report_new)
        df_test_woed[self.y] = self.test[self.y]
        df_oot_woed = bins.whole_woe_replace(self.oot, all_report_new)
        df_oot_woed[self.y] = self.oot[self.y]

        all_report_new["interval"] = all_report_new["Interval"].apply(interval_round)
        all_report_new["Interval"] = all_report_new["Interval"].apply(lambda x: str(x))
        bp = BinPlot(
            df=self.train,
            all_report=all_report_new,
            change_report=change_report,
            y=self.y,
            df_test=self.test,
            df_oot=self.oot,
            bin_num=bin_num
        )
        bp.plot_bin(all_report_new, f"{report_dir}/bin_plot_{self.bin_version}.html")

        # 生成长期趋势图
        cur_selected = change_report["Feature"]
        if self.time_column:
            btp = BinTrendPlot(self.train, cur_selected, self.y, self.time_column, change_report)
            btp.plot_bin(f"{report_dir}/bin_trend_plot_{self.bin_version}.html")


        # 存文件
        df2csv(all_report_new, report_dir, f"all_report_{self.bin_version}.csv")
        df2csv(change_report, report_dir, f"change_report_{self.bin_version}.csv")
        df2csv(df_train_woed, report_dir, "df_train_woed.csv")
        df2csv(df_test_woed, report_dir, "df_test_woed.csv")
        df2csv(df_oot_woed, report_dir, "df_oot_woed.csv")

        # 存pkl
        data_all = {
            "version": self.bin_version,
            "data": {
                "all_report": all_report_new,
                "change_report": change_report,
                "df_train_woed": df_train_woed,
                "df_test_woed": df_test_woed,
                "df_oot_woed": df_oot_woed
            }
        }
        save_to_pkl(data_all, f"{report_dir}/all.pkl")

        # 存更新context
        # self.save_object("all_report", all_report_new)
        # self.save_object("change_report", change_report)
        # self.save_object("df_train_woed", df_train_woed)
        # self.save_object("df_test_woed", df_test_woed)
        # self.save_object("df_oot_woed", df_oot_woed)
        self.save_object("bin_data", data_all)
        self.save_object("bin_plot_html", f"{report_dir}/bin_plot.html")

        # 是否调整x_selected
        if adjust_x:
            chg_x = list(change_report["Feature"].unique())
            self.set_x_selected(chg_x)
            print("x_selected 已经根据change_report改变")

    def binning_on_diff_y(self, y_list: list):
        """
        查看当前的分箱在不同的y上面的表现情况
        :param y_list:
        :return:
        """
        cur_bin_data = self.bin_data
        cur_bin_version = cur_bin_data["version"]
        cur_change_report = cur_bin_data["data"]["change_report"]
        var_list = cur_change_report["Feature"]
        print("当前分箱版本: ", cur_bin_version)
        bdf = BinDiffY(self.train, var_list, y_list, cur_change_report)
        bdf.plot(f"{self.data_dir}/diff_y_{cur_bin_version}.html")

    def binning_version_info(self):
        bin_dir = f"{self.data_dir}/bins/"
        all_versions = os.listdir(bin_dir)
        print(all_versions)

    def binning_from_version(self, bin_version):
        """
        从指定版本的bin,load到当前的context
        :param report_dir:
        :return:
        """
        bin_dir = f"{self.data_dir}/bins/v{bin_version}/"
        print(bin_dir)
        if not os.path.exists(bin_dir):
            print(f"Bin version: {bin_version} not exists, check your version")
            return 1
        if not os.path.exists(f"{bin_dir}/all.pkl"):
            print(f"all.pkl for version: {bin_version} not find!")
            return 1
        data_all = load_from_pkl(f"{bin_dir}/all.pkl")
        this_version = data_all["version"]
        self.bin_version = this_version
        self.save_object("bin_data", data_all)
        print(f"Current context bin_version already change to {this_version} !")

    def generate_report(self, swap_params: dict=None):
        """
        生成模型报告
        :param swap_params:
        :return:
        """
        report = Report(self,
                        self.data_dir,
                        data_dir=self.data_dir,
                        swap_params=swap_params)
        report.generate_report(self.data_dir)
        print("Enjoy!")

    def model_fit(self, X=None, select=None, add_constant=True):
        """
        logit建模
        :param X:
        :param select:
        :return:
        """
        # 版本管理
        # 覆盖当前version, version可以手动设置
        print(f"当前的bin数据的version为: {self.current_bin_data_version}")
        out_dir = f"{self.data_dir}/model/v{self.model_version}/"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        if X is None:
            X = self.x_selected
        x_data = self.get_object("df_train_woed")[X]
        y_data = self.get_object("df_train_woed")[self.y]

        # 检测是否存在完全相关性
        if ModelTools.check_singular_matrix(x_data, y_data):
            print("建模变量存在完全相关性,建模失败, df.corr()?")
            # print(EdaTools.corr_analysis(x_data))
            var2drop = EdaTools.corr_filter(x_data)
            print("以下变量需要drop: ",var2drop)
            return 1

        logit_result = mylogit(x_data, y_data, select=select, add_constant=add_constant)
        if add_constant:
            X = list(logit_result.params.index)
            if "const" in X:
                X.remove("const")
        else:
            X = list(logit_result.params.index)

        if select == "BS":
            drop_record = ModelTools.backward_selected(x_data, y_data)[1]
            pd.DataFrame({"drop_record": drop_record}).to_csv(f"{out_dir}/backward_drop.csv")

        model_summary = ModelTools.ParamEST(logit_result, df_train=x_data[X])


        # 计算lift
        y_pred_train = model_predict(logit_result, x_data[X], add_constant=add_constant)
        y_true_train = y_data
        train_lift = lift_ks(y_true_train,
                             y_pred_train,
                             bins=10,
                             out_dir=out_dir,
                             save_label="train_lift_chart")

        y_pred_test = model_predict(logit_result, self.get_object("df_test_woed")[X], add_constant=add_constant)
        y_true_test = self.get_object("df_test_woed")[self.y]
        test_lift = lift_ks(y_true_test,
                            y_pred_test,
                            bins=10,
                            out_dir=out_dir,
                            save_label="test_lift_chart")

        y_pred_oot = model_predict(logit_result, self.get_object("df_oot_woed")[X], add_constant=add_constant)
        y_true_oot = self.get_object("df_oot_woed")[self.y]
        oot_lift = lift_ks(y_true_oot,
                           y_pred_oot,
                           bins=10,
                           out_dir=out_dir,
                           save_label="oot_lift_chart")

        y_pred_data = {
            "train": {"y_true": y_true_train, "y_pred": y_pred_train},
            "test": {"y_true": y_true_test, "y_pred": y_pred_test},
            "oot": {"y_true": y_true_oot, "y_pred": y_pred_oot},
        }

        # 累计auc
        accum_auc(logit_result, self.df_train_woed, self.y, f"{out_dir}/accum_auc.html")
        # 计算相关性和VIF
        # 相关性
        train_corr = corr_heat_map(self.get_object("df_train_woed"), X)
        train_corr.render(f"{out_dir}/train_corr_{self.model_version}.html")
        # vif
        model_vif = ModelTools.cat_vif(self.get_object("df_train_woed")[X])
        model_vif.to_csv(f"{out_dir}/model_vif_{self.model_version}.csv", index=False, encoding="utf8")

        # 画图
        roc_plot(y_true_train, y_pred_train, f"roc_train_{self.model_version}.png", out_dir)
        roc_plot(y_true_test, y_pred_test, f"roc_test_{self.model_version}.png", out_dir)
        roc_plot(y_true_oot, y_pred_oot, f"roc_oot_{self.model_version}.png", out_dir)

        ks_plot(y_true_train, y_pred_train, f"ks_train_{self.model_version}.png", out_dir)
        ks_plot(y_true_test, y_pred_test, f"ks_test_{self.model_version}.png", out_dir)
        ks_plot(y_true_oot, y_pred_oot, f"ks_oot_{self.model_version}.png", out_dir)

        model_summary.to_excel(f"{out_dir}/model_summary_{self.model_version}.xlsx", sheet_name="summary", index=True)

        df2csv(train_lift, out_dir, f"train_lift_{self.model_version}.csv")
        df2csv(test_lift, out_dir, f"test_lift_{self.model_version}.csv")
        df2csv(oot_lift, out_dir, f"oot_lift_{self.model_version}.csv")
        model_info = {
            "model_version": self.model_version,
            "bin_version": self.current_bin_data_version
        }
        dict2file(model_info, f"{out_dir}/model_info.json")

        # 存档
        # self.save_object("model_result", logit_result)
        # self.save_object("y_pred_data", y_pred_data)

        model_result_data = {
            "model_info": {
                "model_version": self.model_version,
                "bin_version": self.current_bin_data_version
            },
            "model_result": logit_result,
            "y_pred_data": y_pred_data,
            "lift": {
                "train": train_lift,
                "test": test_lift,
                "oot": oot_lift
            }
        }
        self.save_object("model_result_data", model_result_data)

    def model_predict(self, df: pd.DataFrame, y: str):
        """
        按照当前的模型进行打分
        :param df:
        :param y:
        :return:
        """
        cur_model_result = self.model_result_data
        cur_bin_data = self.bin_data

        model_result = cur_model_result["model_result"]
        all_report = cur_bin_data["data"]["all_report"]

        bins = Bins()
        df_woed = bins.whole_woe_replace(df, all_report)
        df_woed[y] = df[y]

        X = list(model_result.params.keys())
        if "const" in X:
            X.remove("const")
        y_pred = model_predict(model_result, df[X], add_constant=True)
        df[f"{y}_score"] = y_pred
        return df

    def fit_raw_data(self, X=None, select=None, add_constant=True):
        # 版本管理
        # 覆盖当前version, version可以手动设置
        print(f"当前的bin数据的version为: {self.current_bin_data_version}")
        out_dir = f"{self.data_dir}/model/v{self.model_version}/"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        if X is None:
            X = self.x_selected
        x_data = self.get_object("train")[X]
        y_data = self.get_object("train")[self.y]

        # 检测是否存在完全相关性
        if ModelTools.check_singular_matrix(x_data, y_data):
            print("建模变量存在完全相关性,建模失败, df.corr()?")
            # print(EdaTools.corr_analysis(x_data))
            var2drop = EdaTools.corr_filter(x_data)
            print("以下变量需要drop: ", var2drop)
            return 1

        logit_result = mylogit(x_data, y_data, select=select, add_constant=add_constant)
        if add_constant:
            X = list(logit_result.params.index)
            if "const" in X:
                X.remove("const")
        else:
            X = list(logit_result.params.index)

        if select == "BS":
            drop_record = ModelTools.backward_selected(x_data, y_data)[1]
            pd.DataFrame({"drop_record": drop_record}).to_csv(f"{out_dir}/backward_drop.csv")

        model_summary = ModelTools.ParamEST(logit_result, df_train=x_data[X])

        # 计算lift
        y_pred_train = model_predict(logit_result, x_data[X], add_constant=add_constant)
        y_true_train = y_data
        train_lift = lift_ks(y_true_train,
                             y_pred_train,
                             bins=10,
                             out_dir=out_dir,
                             save_label="train_lift_chart")

        y_pred_test = model_predict(logit_result, self.get_object("test")[X], add_constant=add_constant)
        y_true_test = self.get_object("test")[self.y]
        test_lift = lift_ks(y_true_test,
                            y_pred_test,
                            bins=10,
                            out_dir=out_dir,
                            save_label="test_lift_chart")

        y_pred_oot = model_predict(logit_result, self.get_object("oot")[X], add_constant=add_constant)
        y_true_oot = self.get_object("oot")[self.y]
        oot_lift = lift_ks(y_true_oot,
                           y_pred_oot,
                           bins=10,
                           out_dir=out_dir,
                           save_label="oot_lift_chart")

        y_pred_data = {
            "train": {"y_true": y_true_train, "y_pred": y_pred_train},
            "test": {"y_true": y_true_test, "y_pred": y_pred_test},
            "oot": {"y_true": y_true_oot, "y_pred": y_pred_oot},
        }

        # 累计auc
        accum_auc(logit_result, self.df_train_woed, self.y, f"{out_dir}/accum_auc.html")
        # 计算相关性和VIF
        # 相关性
        train_corr = corr_heat_map(self.get_object("train"), X)
        train_corr.render(f"{out_dir}/train_corr_{self.model_version}.html")
        # vif
        model_vif = ModelTools.cat_vif(self.get_object("train")[X])
        model_vif.to_csv(f"{out_dir}/model_vif_{self.model_version}.csv", index=False, encoding="utf8")

        # 画图
        roc_plot(y_true_train, y_pred_train, f"roc_train_{self.model_version}.png", out_dir)
        roc_plot(y_true_test, y_pred_test, f"roc_test_{self.model_version}.png", out_dir)
        roc_plot(y_true_oot, y_pred_oot, f"roc_oot_{self.model_version}.png", out_dir)

        ks_plot(y_true_train, y_pred_train, f"ks_train_{self.model_version}.png", out_dir)
        ks_plot(y_true_test, y_pred_test, f"ks_test_{self.model_version}.png", out_dir)
        ks_plot(y_true_oot, y_pred_oot, f"ks_oot_{self.model_version}.png", out_dir)

        model_summary.to_excel(f"{out_dir}/model_summary_{self.model_version}.xlsx", sheet_name="summary", index=True)

        df2csv(train_lift, out_dir, f"train_lift_{self.model_version}.csv")
        df2csv(test_lift, out_dir, f"test_lift_{self.model_version}.csv")
        df2csv(oot_lift, out_dir, f"oot_lift_{self.model_version}.csv")
        model_info = {
            "model_version": self.model_version,
            "bin_version": self.current_bin_data_version
        }
        dict2file(model_info, f"{out_dir}/model_info.json")

        model_result_data = {
            "model_info": {
                "model_version": self.model_version,
                "bin_version": self.current_bin_data_version
            },
            "model_result": logit_result,
            "y_pred_data": y_pred_data,
            "lift": {
                "train": train_lift,
                "test": test_lift,
                "oot": oot_lift
            }
        }
        self.save_object("model_result_data", model_result_data)
