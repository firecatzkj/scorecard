# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:08:26 2019

@author: zhaifeifei1
"""

# %%

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc
import xlwt
from xlwt import *

# %%

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''          七、auc ks lift swap分析自动化函数                '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


## 计算K-S
def cal_ks(point, Y, section_num=10):
    Y = pd.Series(Y)
    sample_num = len(Y)

    bad_percent = np.zeros([section_num, 1])
    good_percent = np.zeros([section_num, 1])

    point = pd.DataFrame(point)
    sorted_point = point.sort_values(by=0, ascending=False)
    total_bad_num = len(np.where(Y == 1)[0])
    total_good_num = len(np.where(Y == 0)[0])

    for i in range(0, section_num):
        split_point = sorted_point.iloc[int(round(sample_num * (i + 1) / section_num)) - 1]
        position_in_this_section = np.where(point >= split_point)[0]
        bad_percent[i] = len(np.where(Y.iloc[position_in_this_section] == 1)[0]) / total_bad_num
        good_percent[i] = len(np.where(Y.iloc[position_in_this_section] == 0)[0]) / total_good_num

    ks_value = np.abs(bad_percent - good_percent)

    return ks_value, bad_percent, good_percent


## 创建样式
borders = xlwt.Borders()
borders.left = xlwt.Borders.THIN
borders.right = xlwt.Borders.THIN
borders.top = xlwt.Borders.THIN
borders.bottom = xlwt.Borders.THIN

style = xlwt.XFStyle()  # 创建样式
font = xlwt.Font()
alignment = xlwt.Alignment()
alignment.horz = 0x02
alignment.vert = 0x01
style.alignment = alignment
style.borders = borders
font.name = u'宋体'  # 字体为Arial
style.font.name = font.name
style.font.height = 220  # 设置字体大小(220为11号字体，间隔40为一个字体)

style1 = xlwt.XFStyle()
alignment = xlwt.Alignment()
alignment.horz = 0x02
alignment.vert = 0x01
style1.alignment = alignment
style1.borders = borders
pattern = xlwt.Pattern()  # Create the pattern
pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
pattern.pattern_fore_colour = 5
style1.pattern = pattern
font = xlwt.Font()
font.name = u'宋体'
style1.font.name = font.name
style1.font.height = 220

style2 = xlwt.XFStyle()  # 创建样式
font = xlwt.Font()
alignment = xlwt.Alignment()
alignment.horz = 0x02
alignment.vert = 0x01
style2.alignment = alignment
style2.borders = borders
font.name = u'宋体'  # 字体为Arial
style2.font.name = font.name
style2.font.height = 220  # 设置字体大小(220为11号字体，间隔40为一个字体)
style2.num_format_str = '#0.00%'

style4 = xlwt.XFStyle()
alignment = xlwt.Alignment()
alignment.horz = 0x02
alignment.vert = 0x01
style4.alignment = alignment
style4.borders = borders
font = xlwt.Font()
font.name = u'宋体'
pattern = xlwt.Pattern()  # Create the pattern
pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
pattern.pattern_fore_colour = 172
style4.pattern = pattern
font.bold = True
style4.font.bold = font.bold
style4.font.name = font.name
style4.font.height = 220


## AucKsExcel样式
def AucKsExcelFormat(sheet, row_start_point, col_start_point, table_name, P1_name, P2_name, corr):
    sheet.write(row_start_point, col_start_point, table_name, style1)
    sheet.write(row_start_point + 1, col_start_point, "AUC", style4)
    sheet.write(row_start_point + 2, col_start_point, "K-S", style4)
    sheet.write(row_start_point + 3, col_start_point, "相关性", style4)
    sheet.write(row_start_point, col_start_point + 1, P1_name, style4)
    sheet.write(row_start_point, col_start_point + 2, P2_name, style4)
    sheet.write_merge(row_start_point + 3, row_start_point + 3, col_start_point + 1, col_start_point + 2, corr, style)


## RankExcel样式
def RankExcelFormat(sheet, row_start_point, col_start_point, table_name, P1_name, P2_name, GroupNums=10):
    sheet.write(row_start_point, col_start_point, table_name, style1)
    sheet.write(row_start_point, col_start_point + 1, P1_name, style4)
    sheet.write(row_start_point, col_start_point + 2, P2_name, style4)
    for i in range(GroupNums):
        sheet.write(row_start_point + 1 + i, col_start_point, i + 1, style4)


## SwapExcel样式
def SwapExcelFormat(sheet, row_start_point, col_start_point, table_name, P1_name, P2_name, GroupNums1, GroupNums2):
    col1 = sheet.col(0)
    col1.width = 200 * 20
    col2 = sheet.col(GroupNums2 + 3)
    col2.width = 200 * 20

    sheet.write(row_start_point, col_start_point, table_name, style1)
    sheet.write(row_start_point + 1, col_start_point, P1_name, style4)
    sheet.write_merge(row_start_point, row_start_point, col_start_point + 1, col_start_point + GroupNums2, P2_name,
                      style4)
    for i in range(GroupNums1):
        sheet.write(row_start_point + 2 + i, col_start_point, i + 1, style4)
    for i in range(GroupNums2):
        sheet.write(row_start_point + 1, col_start_point + 1 + i, i + 1, style4)
    sheet.write(row_start_point + GroupNums1 + 2, col_start_point, "总计", style4)
    sheet.write(row_start_point + 1, col_start_point + GroupNums2 + 1, "总计", style4)


#    SwapExcelFormat(sheet1,18,0,"逾期率",P1_name,P2_name,GroupNums1,GroupNums2)

## Swap-in-out样式
def SwapInOutExcelFormat(sheet, row_start_point, col_start_point, P1_name, P2_name, CutNumP1, CutNumP2):
    sheet.write_merge(row_start_point, row_start_point + 1, col_start_point, col_start_point, "swap", style1)
    merge_name = "{0}前{1}组与{2}前{3}组汇总分析".format(P1_name, CutNumP1, P2_name, CutNumP2)
    sheet.write_merge(row_start_point, row_start_point, col_start_point + 1, col_start_point + 6, merge_name, style4)
    col_label = ['label', 'tot', 'bad', 'per_rate', 'bad_rate', 'bad_rate_acu']
    for i in range(len(col_label)):
        col = col_label[i]
        sheet.write(row_start_point + 1, col_start_point + 1 + i, col, style4)
    for i in range(1, 5):
        sheet.write(row_start_point + 1 + i, col_start_point, i, style4)


## 等分十段，给每个人打上打上分组标签
def RankLabel(Pvalue, Numcutoff=10):
    Pvalue_unique = np.unique(Pvalue)
    splitpoint = list(pd.qcut(Pvalue_unique, Numcutoff, retbins=True)[-1])
    Plabel = Pvalue.copy()
    eps = 1e-06
    for i in range(1, Numcutoff + 1):
        if i == 1:
            position_in_this_section = np.where(Pvalue < splitpoint[i] + eps)[0]
        else:
            position_in_this_section = np.where((Pvalue > splitpoint[i - 1]) & (Pvalue < splitpoint[i] + eps))[0]
        Plabel.iloc[position_in_this_section] = i
    Plabel = Plabel.astype(int)

    return splitpoint, Plabel


## swap分析
def SwapData(df, y, P1, P2, GroupNums1, GroupNums2):
    #    df =df_swap.copy()
    P1_splitpoint, P1_label_tmp = RankLabel(df[P1], Numcutoff=GroupNums1)
    P2_splitpoint, P2_label_tmp = RankLabel(df[P2], Numcutoff=GroupNums2)

    df_tmp = pd.concat([df.loc[:, [P1, P2, y]], pd.DataFrame(P1_label_tmp.values, columns=['P1_label']),
                        pd.DataFrame(P2_label_tmp.values, columns=['P2_label'])], axis=1)

    P1_num_bad_count = pd.pivot_table(df_tmp, index=['P1_label'], values=y, aggfunc=[len, np.sum])
    P2_num_bad_count = pd.pivot_table(df_tmp, index=['P2_label'], values=y, aggfunc=[len, np.sum])

    df_swap_len_tmp = pd.pivot_table(df_tmp, index=['P1_label'], columns=['P2_label'], values=y, aggfunc=[len])

    ## 增加汇总行和列
    df_swap_len_tmp = pd.DataFrame(df_swap_len_tmp.values, index=range(len(df_swap_len_tmp)),
                                   columns=range(df_swap_len_tmp.shape[1]))
    tmp_col = pd.DataFrame(df_swap_len_tmp.sum(axis=1), columns=[df_swap_len_tmp.shape[1]])
    df_swap_len = df_swap_len_tmp.join(tmp_col)
    tmp_row = pd.DataFrame(df_swap_len_tmp.sum(axis=0), columns=[df_swap_len_tmp.shape[0]]).T
    df_swap_len = df_swap_len.append(tmp_row)
    df_swap_len.iloc[df_swap_len.shape[0] - 1, df_swap_len.shape[1] - 1] = len(df)

    df_swap_sum_tmp = pd.pivot_table(df_tmp, index=['P1_label'], columns=['P2_label'], values=y, aggfunc=[np.sum])
    ## 增加汇总行和列
    df_swap_sum_tmp = pd.DataFrame(df_swap_sum_tmp.values, index=range(len(df_swap_sum_tmp)),
                                   columns=range(df_swap_len_tmp.shape[1]))
    tmp_col = pd.DataFrame(df_swap_sum_tmp.sum(axis=1), columns=[df_swap_sum_tmp.shape[1]])
    df_swap_sum = df_swap_sum_tmp.join(tmp_col)
    tmp_row = pd.DataFrame(df_swap_sum_tmp.sum(axis=0), columns=[df_swap_sum_tmp.shape[0]]).T
    df_swap_sum = df_swap_sum.append(tmp_row)
    df_swap_sum.iloc[df_swap_sum.shape[0] - 1, df_swap_sum.shape[1] - 1] = sum(df[y])

    ##为缺失分组添加一个样本：
    df_swap_len = df_swap_len.fillna(1)
    df_swap_sum = df_swap_sum.fillna(0)

    df_swap_len_rate = pd.DataFrame(df_swap_len.values / len(df))

    df_swap_sum_rate = pd.DataFrame(df_swap_sum.values / df_swap_len.values)

    return P1_num_bad_count, P2_num_bad_count, df_swap_len, df_swap_sum, df_swap_len_rate, df_swap_sum_rate


## swapinout data
def SwapInOutData(df_swap_len, df_swap_sum, CutNumP1, CutNumP2, GroupNums1, GroupNums2):
    df_tmp = pd.DataFrame(['in_in', 'swap_in', 'swap_out', 'out_out'])
    len_in_in_data = np.array(df_swap_len.iloc[:CutNumP1, :CutNumP2]).sum()
    len_swap_in_data = np.array(df_swap_len.iloc[:CutNumP1, CutNumP2:GroupNums2]).sum()
    len_swap_out_data = np.array(df_swap_len.iloc[CutNumP1:GroupNums1, :CutNumP2]).sum()
    len_out_out_data = np.array(df_swap_len.iloc[CutNumP1:GroupNums1, CutNumP2:GroupNums2]).sum()

    sum_in_in_data = np.array(df_swap_sum.iloc[:CutNumP1, :CutNumP2]).sum()
    sum_swap_in_data = np.array(df_swap_sum.iloc[:CutNumP1, CutNumP2:GroupNums2]).sum()
    sum_swap_out_data = np.array(df_swap_sum.iloc[CutNumP1:GroupNums1, :CutNumP2]).sum()
    sum_out_out_data = np.array(df_swap_sum.iloc[CutNumP1:GroupNums1, CutNumP2:GroupNums2]).sum()

    tot = pd.Series([len_in_in_data, len_swap_in_data, len_swap_out_data, len_out_out_data])
    bad = pd.Series([sum_in_in_data, sum_swap_in_data, sum_swap_out_data, sum_out_out_data])
    bad_cnt = pd.Series([bad[:1].sum(), bad[:2].sum(), bad[:3].sum(), bad[:4].sum()])

    per_rate = tot / sum(tot)
    bad_rate = bad / tot
    bad_rate_acu = bad_cnt / sum(tot)

    df_tot_bad = pd.concat([df_tmp, tot, bad], axis=1)
    df_tot_bad.columns = range(df_tot_bad.shape[1])

    df_tot_bad_rate = pd.concat([per_rate, bad_rate, bad_rate_acu], axis=1)

    return df_tot_bad, df_tot_bad_rate


def WriteData(df, sheet, row_start_point, col_start_point):
    for col in range(df.shape[1]):
        tmp = df[df.columns[col]]
        for row in range(df.shape[0]):
            try:
                sheet.write(row_start_point + row, col_start_point + col, tmp[row].astype(float), style)
            except:
                sheet.write(row_start_point + row, col_start_point + col, tmp[row], style)


def WriteData2(df, sheet, row_start_point, col_start_point, GroupNums1, GroupNums2):
    """
    df =Rank_bad.copy()
    sheet =sheet1
    row_start_point =2
    col_start_point =5
    """
    for row in range(GroupNums1):
        tmp = df.iloc[:, 0]
        try:
            sheet.write(row_start_point + row, col_start_point, tmp[row].astype(float), style2)
        except:
            sheet.write(row_start_point + row, col_start_point, tmp[row], style2)

    for row in range(GroupNums2):
        tmp = df.iloc[:, 1]
        try:
            sheet.write(row_start_point + row, col_start_point + 1, tmp[row].astype(float), style2)
        except:
            sheet.write(row_start_point + row, col_start_point + 1, tmp[row], style2)


def WriteData3(df, sheet, row_start_point, col_start_point):
    for col in range(df.shape[1]):
        tmp = df[df.columns[col]]
        for row in range(df.shape[0]):
            try:
                sheet.write(row_start_point + row, col_start_point + col, tmp[row].astype(float), style2)
            except:
                sheet.write(row_start_point + row, col_start_point + col, tmp[row], style2)


# %%

def Swap(df, y, P1, P2, P1_name, P2_name, CutNumP1, CutNumP2, filename, outputh, GroupNums1=10, GroupNums2=10):
    """
    df =df_swap.copy()
    filename ="test3.xls"
    GroupNums1 =10
    GroupNums2 =3
    """
    ks_value, bad_percent, good_percent = cal_ks(np.array(df[P1]), np.array(df[y]), section_num=10)
    false_positive_rate, recall, thresholds = roc_curve(df[y], df[P1])
    roc_auc = auc(false_positive_rate, recall)
    P1_ks = np.max(ks_value)
    P1_auc = roc_auc

    ks_value, bad_percent, good_percent = cal_ks(np.array(df[P2]), np.array(df[y]), section_num=10)
    false_positive_rate, recall, thresholds = roc_curve(df[y], df[P2])
    roc_auc = auc(false_positive_rate, recall)
    P2_ks = np.max(ks_value)
    P2_auc = roc_auc

    P1_rank, P2_rank, df_swap_len, df_swap_sum, df_swap_len_rate, df_swap_sum_rate = SwapData(df, y, P1, P2, GroupNums1,
                                                                                              GroupNums2)

    ##计算分组坏账率
    P1_rank_bad = P1_rank.iloc[:, 1] / P1_rank.iloc[:, 0]
    P2_rank_bad = P2_rank.iloc[:, 1] / P2_rank.iloc[:, 0]
    Rank_bad = pd.concat([pd.DataFrame(P1_rank_bad.values, columns=['P1_rank_bad']),
                          pd.DataFrame(P2_rank_bad.values, columns=['P2_rank_bad'])], axis=1)

    book = xlwt.Workbook(encoding='utf-8')
    sheet1 = book.add_sheet("交叉分析")

    ##auc ks rank写入
    corr = np.corrcoef(df[P1], df[P2])[0, 1]
    AucKsExcelFormat(sheet1, 1, 0, "模型效果", P1_name, P2_name, corr.round(4))
    sheet1.write(2, 1, P1_auc.round(4), style)
    sheet1.write(2, 2, P2_auc.round(4), style)
    sheet1.write(3, 1, P1_ks.round(4), style)
    sheet1.write(3, 2, P2_ks.round(4), style)

    ## Rank写入
    RankExcelFormat(sheet1, 1, 4, "评分排序", P1_name, P2_name, GroupNums=10)
    WriteData2(Rank_bad, sheet1, 2, 5, GroupNums1, GroupNums2)

    ## Swap写入
    SwapExcelFormat(sheet1, 18, 0, "逾期率", P1_name, P2_name, GroupNums1, GroupNums2)
    WriteData3(df_swap_sum_rate, sheet1, 20, 1)

    SwapExcelFormat(sheet1, 18, GroupNums2 + 3, "人数占比", P1_name, P2_name, GroupNums1, GroupNums2)
    WriteData3(df_swap_len_rate, sheet1, 20, GroupNums2 + 4)

    SwapExcelFormat(sheet1, 30 + GroupNums1, 0, "总体人数分布", P1_name, P2_name, GroupNums1, GroupNums2)
    WriteData(df_swap_len, sheet1, 32 + GroupNums1, 1)

    SwapExcelFormat(sheet1, 30 + GroupNums1, GroupNums2 + 3, "逾期人数分布", P1_name, P2_name, GroupNums1, GroupNums2)
    WriteData(df_swap_sum, sheet1, 32 + GroupNums1, GroupNums2 + 4)

    ## swapinout data 写入
    df_tot_bad, df_tot_bad_rate = SwapInOutData(df_swap_len, df_swap_sum, CutNumP1, CutNumP2, GroupNums1, GroupNums2)
    SwapInOutExcelFormat(sheet1, 23 + GroupNums1, 0, P1_name, P2_name, CutNumP1, CutNumP2)
    WriteData(df_tot_bad, sheet1, 25 + GroupNums1, 1)
    WriteData3(df_tot_bad_rate, sheet1, 25 + GroupNums1, 4)

    ## 文件输出
    filename = filename
    outputh = outputh
    book.save(outputh + filename)
