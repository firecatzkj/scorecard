# -*- coding: utf8 -*-
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:38:18 2019

@author: clay
"""
import os
import pandas as pd
os.chdir("/Users/clay/pyscorecard")
from pyscorecard.context import DataContext
from pyscorecard.tools import split_data

# Step1: 数据准备
# 每个人处理数据的方式不同,这里仅仅作为示例

def prepare():
    df = pd.read_csv(r"data/qhxlj.csv")
    df_train, df_test = split_data(df)
    df_oot = pd.read_csv(r"data/qhyzj.csv")
    df_train["add_mon"] = df_train["add_day"][0:7]
    df_test["add_mon"] = df_test["add_day"][0:7]
    df_oot["add_mon"] = df_oot["add_day"][0:7]
    var_types_map = {
        "loan_use": "M",
        "liability_status": "M",
        "credit_status": "M",
        "industry_type": "M",
        "family_live_type": "M",
        "user_marriage": "M",
        "position": "M",
        "household_type": "M",
        "company_type": "M",
        "contact1a_relationship": "M"
    }
    return df_train, df_test, df_oot, var_types_map

df_train, df_test, df_oot, var_types_map = prepare()

# Step2: 创建context(管理建模过程中产生的中间变量)
# 新版代码把data_dir也放到初始化的函数里面了
data_dir = "./data/"

dc = DataContext(y="y",
                 train=df_train,
                 test=df_test,
                 oot=df_oot,
                 dtypes=var_types_map,
                 data_dir=data_dir)

# Step3: Eda
# eda由于用了multiprocessing,如果长时间跑不出来,尝试加上foreze_support()
# Step3.1: 三个数据集的eda
dc.eda_report_all()

# Step3.2: 指定某一个数据集的eda
dc.eda_filter(dataset="train", corr=0.85, missing=0.95, spar=0.95, uniq=0.99)

# Step3.3: 通过eda筛选变量(可选)
# help(dc.eda_filter)  查看更多的帮助信息
dc.eda_filter(missing=0.95)

# Step4: 变量选择
# 单报告
dc.feature_report(x=None, data_set="train")
# 多报告
dc.feature_report_all()

# Stepx: 变量筛选管理
# 从x_selected删除/增加一个变量
dc.drop_x("理由", "变量名")
dc.add_x()
# 重置x_selected
dc.set_x_selected()


# Step5: 分箱
# 初始化分箱: 单调分箱
dc.bin_version = "do_mono_v1"
dc.binning_init(MDP=True, end_num=6, mono=True, method=3)

# 初始化分箱: 不单调分箱
dc.bin_version = "not_mono"
dc.binning_init(MDP=True, end_num=6, mono=False, method=3)

# test
dc.bin_version = "plot_change"
dc.binning_init(MDP=True, end_num=6, mono=False, method=3)


# 一个拐点
dc.bin_version = "one_jump_point"
dc.binning_init(MDP=True, end_num=6, mono=True, method=3, jn=1)

# bin_plot_debug
dc.bin_version = "bin_plot_debug"
dc.binning_init(data_dir, MDP=True, end_num=6, mono=False, method=3)


# 载入某一个bin的版本
dc.binning_from_version("do_mono_v1", data_dir)

# 手动修改分箱之后重新生成相关文件
change_report = pd.read_csv("xxxxxxxxx/change_report_not_mono.csv")
dc.bin_version = "manual_bin_v4"
dc.binning_from_change(change_report, adjust_x=True)


dc.set_x_selected(list(var_types_map.keys()))

# Step6: 建模
dc.model_version = "no_select"
dc.model_fit(X=None, select=None)

dc.model_version = "BS"
dc.model_fit(X=None, select="BS")


dc.model_version = "FS"
dc.model_fit(X=None, select="FS")


dc.model_version = "add_vif_corr"
dc.model_fit(X=None, select="BS")


dc.model_version = "report_debug"
dc.model_fit(X=None, select=None)
    

# Step7: 生成报告
# 如果分箱,和建模都做了
# 不带swap分析的
dc.generate_report()
# 带swap分析的
swap_params = {
        "df": "一个csv",
        "y": "y",
        "P1": "model1_score",
        "P2": "model2_score",
        "GroupNums1": 10,
        "GroupNums2": 10
        }
dc.generate_report(swap_params=swap_params)


# Have Fun!    ^_^
dc.bin_data["data"]["df_train_woed"].corr()