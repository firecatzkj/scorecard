# -*- coding: utf8 -*-
import math
import pandas as pd
import numpy as np
from scipy import stats
from copy import copy
from sklearn import tree
from functools import reduce


pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 2500)


class BinningInfo:
    """
    通用分箱信息整理
    只需填入初始方法的参数,就能计算出对应的信息
    """
    def __init__(self, df: pd.DataFrame, x: str, y: str, cutpoint: list):
        """
        分箱信息整理评估
        :param df: 数据集
        :param x: 变量
        :param y: Y
        :param cutpoint: cutpoint
        """
        self.df = df
        self.x = x
        self.y = y
        self.cutpoint = cutpoint

    @staticmethod
    def calc_bump(badrate: pd.Series) -> int:
        """
        变量分箱趋势量化: 计算bump点
        可行性待讨论
        :param badrate:
        :return: bump个数
        """
        bump = 0
        trend = stats.linregress(list(range(len(badrate))), badrate).slope

        if trend > 0:
            for i in range(1, len(badrate), 1):
                if badrate[i] < badrate[i - 1]:
                    bump += 1
                else:
                    continue
        elif trend < 0:
            for i in range(1, len(badrate), 1):
                if badrate[i] > badrate[i - 1]:
                    bump += 1
        return bump

    @staticmethod
    def calc_columns(single_result, tmp, Y, df):
        single_result["total"] = len(tmp)
        single_result["good"] = len(tmp[tmp[Y] == 0])
        single_result["bad"] = len(tmp[tmp[Y] == 1])
        single_result["goodDistr"] = round((single_result["good"] / len(df[df[Y] == 0])), 6) if len(
            df[df[Y] == 0]) != 0 else 0
        single_result["badDistr"] = round((single_result["bad"] / len(df[df[Y] == 1])), 6) if len(
            df[df[Y] == 1]) != 0 else 0
        single_result["distr"] = round((len(tmp) / len(df)), 6)
        single_result["badRate"] = round((single_result["bad"] / single_result["total"]), 6) if single_result[
                                                                                                    "total"] != 0 else 0
        single_result["Odds"] = round((single_result["bad"] / single_result["good"]), 6) if single_result[
                                                                                                "good"] != 0 else 0
        if (single_result["badDistr"] == 0) or (single_result["goodDistr"] == 0):
            single_result["WOE"] = 0
        else:
            single_result["WOE"] = round(math.log(single_result["badDistr"] / single_result["goodDistr"]), 6)
        single_result["IV"] = round((single_result["badDistr"] - single_result["goodDistr"]) * single_result["WOE"], 4)

        return copy(single_result)

    def combin_bins(self):
        """
        合并异常区间
        :return:
        """
        pass

    @staticmethod
    def binning_manual(df: pd.DataFrame, Y: str, x: str, cuts: list) -> pd.DataFrame:
        """
        手动binning
        :param df:
        :param Y:
        :param x:
        :param cuts:
        :return:
        """
        # 排序很重要,因为后面算区间的时候需要索引+1 -1
        cuts = [float("-inf"), ] + list(cuts) + [float("+inf"), ]
        cuts.sort()
        single_sample = df[[Y, x]]
        single_sample = pd.DataFrame(single_sample)
        result = []
        base_single_result = dict.fromkeys([
            "cutpoints",
            "total",
            "good",
            "bad",
            "goodDistr",
            "badDistr",
            "distr",
            "badRate",
            "Odds",
            "WOE",
            "IV"
        ], None)

        for i in range(1, len(cuts), 1):
            single_result = copy(base_single_result)
            one_cut = pd.Interval(cuts[i - 1], cuts[i], closed="right")
            tmp = single_sample.dropna()
            single_result["cutpoints"] = "{}".format(one_cut)
            tmp = tmp[
                (tmp[x] > one_cut.left) &
                (tmp[x] <= one_cut.right)]
            result.append(BinningInfo.calc_columns(single_result, tmp, Y, df))

        # 计算missing和total
        # missing
        tmp_missing = single_sample[pd.isnull(single_sample[x])]
        single_result_missing = BinningInfo.calc_columns(copy(base_single_result), tmp_missing, Y, df)
        single_result_missing["cutpoints"] = "Missing"
        result.append(single_result_missing)
        # total
        result = pd.DataFrame(result)
        tmp_total = single_sample
        single_result_total = BinningInfo.calc_columns(copy(base_single_result), tmp_total, Y, df)
        single_result_total["cutpoints"] = "Total"
        single_result_total["IV"] = round(result["IV"].sum(), 6)
        result = result.append(pd.DataFrame([single_result_total, ]))[[
            "cutpoints", "total", "good", "bad", "goodDistr", "badDistr", "distr", "badRate", "Odds", "WOE", "IV"]]
        return result

    def bininfo(self):
        """
        计算分箱信息
        :return:
        """

        res = self.binning_manual(
            self.df,
            self.y,
            self.x,
            self.cutpoint
        )
        return res

    def iv(self):
        """
        计算IV
        :return:
        """
        return list(self.bininfo()["IV"])[-1]

    def bump(self):
        """
        计算分箱区间的跳点
        :return:
        """
        return self.calc_bump(self.bininfo()["badRate"][: -2])


class BinningTools:

    @staticmethod
    def assign_group(x, split):
        N = len(split)
        if x <= min(split):
            return min(split)
        elif x > max(split):
            return np.inf
        else:
            for i in range(N - 1):
                if split[i] < x <= split[i + 1]:
                    return split[i + 1]

    @staticmethod
    def Chi2(data, total, bad, allbadrate):
        '''
        ###计算相邻两组的卡方值公式###
        : param data: dataframe 待出来的dataframe
        : param total： str 总体样本
        : param bad： str 坏样本
        : param bad： allbadrate 坏账率
        : return Chi2：float 卡方值
        '''
        data = data.copy()
        data['expected'] = data[total].apply(lambda x: x * allbadrate)
        combined = zip(data[bad], data['expected'])
        chi = [(i[1] - i[0]) ** 2 / i[1] for i in combined if i[1] != 0]
        Chi2 = sum(chi)
        return Chi2

    @staticmethod
    def calc_chi(df, col, target):
        """
        ###计算分箱后的badrate,pcnt，卡方值和单调性检验前要用到的数据###
        : param df: dataframe 待出来的dataframe
        : param col: str 待出来的列
        : param target： str 样本标签
        : return regroup：dataframe
        : return regroup：chisqlist
        """
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total': total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(inplace=True)
        # regroup['badrate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total * 1.0, axis=1)
        regroup['badrate'] = regroup.apply(lambda x: x["bad"] * 1.0 / x["total"] * 1.0, axis=1)
        # regroup['badrate'] = regroup["bad"] / regroup["total"]

        regroup = regroup.sort_values([col])
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        regroup['Pcnt'] = regroup.apply(lambda x: x.total * 1.0 / N, axis=1)
        # regroup['Pcnt'] = regroup["total"] / N

        allBadRate = B * 1.0 / N * 1.0
        cutOffPoint = [[i] for i in regroup[col]]
        chisqlist = []
        for k in range(len(cutOffPoint) - 1):
            temp_colList = cutOffPoint[k] + cutOffPoint[k + 1]
            temp_regroup = regroup.loc[regroup[col].isin(temp_colList)]
            chi = BinningTools.Chi2(temp_regroup, 'total', 'bad', allBadRate)
            chisqlist.append(chi)
        return regroup, chisqlist, allBadRate

    @staticmethod
    def cut_str(x: pd.Series, bins: list, labels: list=None):
        """
        字符串pd.cut
        :param x:
        :param bins:
        :param labels:
        :return:
        """
        bins = list(map(lambda x: list(x), bins))
        if labels is None:
            labels = list(map(
                lambda z: f"b{z}",
                range(len(bins))
            ))
        tmp_map = {k: v for k, v in zip(labels, bins)}
        print(tmp_map, ">>>>>>")
        bin_map = {}
        for k, v in tmp_map.items():
            for i in v:
                bin_map[i] = k
        all_category = reduce(lambda x, y: list(x) + list(y), bins)

        def inner_trans(x):
            if x in bin_map.keys():
                return bin_map[x]
            else:
                return None
        print(bin_map)
        print(all_category)
        return pd.Series(x).apply(inner_trans)




class Binning:
    """
    分箱算法集合
    统一返回cutpoint
    """

    @staticmethod
    def binning_tree(df, Y, x, num_max_bins=10) -> list:
        """
        决策树分箱, single x ~ y
        :param df:
        :param Y:
        :param x:
        :param num_max_bins:
        :return:
        """
        df = df[[x, Y]].dropna()
        y = df[Y]
        x_test = df[[x, ]]
        mytree = tree.DecisionTreeClassifier(
            max_features=1,
            min_weight_fraction_leaf=0.05,
            criterion="entropy",
            max_leaf_nodes=num_max_bins)
        mytree.fit(x_test, y)
        cutpoint = mytree.tree_.threshold
        cutpoint = cutpoint[cutpoint != -2]
        cutpoint.sort()
        return cutpoint

    @staticmethod
    def binning_chi(df, col, target, max_intervals):
        '''
        ###卡方分箱函数###
        : param df: dataframe 待出来的dataframe
        : param col: str 待出来的列
        : param target： str 样本标签
        : param max_intervals: float 最大分组个数
        : return groupIntervalst：list 输出切分点
        '''
        df2 = df.copy()
        colLevel = list(set(df[col]))
        N_distinct = len(colLevel)
        if N_distinct > 50:
            split_x = Binning.binning_equidistant(df2, col, 50)
            df2[col] = df2[col].map(lambda x: BinningTools.assign_group(x, split_x))
        chisqList = BinningTools.calc_chi(df2, col, target)[1]
        groupIntervals = sorted(list(set(df2[col])))
        while (len(groupIntervals) > max_intervals):
            minChiIndex = chisqList.index(min(chisqList))
            groupIntervals.remove(groupIntervals[minChiIndex])
            df2[col] = df2[col].map(lambda x: BinningTools.assign_group(x, groupIntervals))
            chisqList = BinningTools.calc_chi(df2, col, target)[1]
            groupIntervals = sorted(list(set(df2[col])))
        return groupIntervals

    @staticmethod
    def binning_equidistant(df, col, max_intervals):
        '''
        ###等距分箱函数 ###
        : param df: dataframe 待出来的dataframe
        : param col: str 待出来的列
        : param max_intervals： int 最大的分组数
        : return splitPoint：list 分组切分点
        '''
        var_max, var_min = max(df[col]), min(df[col])
        interval_len = (var_max - var_min) * 1.0 / max_intervals
        splitPoint = [var_min + i * interval_len for i in range(1, max_intervals)]
        splitPoint.sort()
        return splitPoint

    @staticmethod
    def binning_frequency(df, col, max_intervals, dtype="C"):
        """
        : param df: dataframe 待出来的dataframe
        : param col: str 待出来的列
        : param max_intervals： int 最大的分组数
        : return splitPoint：list 分组切分点
        """
        print(col, dtype, "SSSSSSSSSSSSSSSS")
        if dtype == "M":
            return list(df[col].unique())
        else:
            percent = [i / max_intervals for i in range(1, max_intervals, 1)]
            splitPoint = df[col].quantile(percent).values.tolist()
            splitPoint = list(set(splitPoint))
            splitPoint.sort(reverse=False)
            return splitPoint

    @staticmethod
    def binning_category(df, col):
        """
        对类别型变量进行分箱
        :param df:
        :param col:
        :return:
        """
        return list(df[col].unique())

    # @staticmethod
    # def merge_mono(df, col, cutOffPoint, target):
    #     ###该函数主要解决分值之后分组单调问题###
    #     '''
    #     : param df: dataframe 待处理的dataframe
    #     : param col: str 待出来的列
    #     : param cutOffPoint: list 这一列的切分点
    #     : param target： str 样本标签
    #     : return cutOffPoint：list 输出切分点
    #     '''
    #     df[col] = df[col].apply(lambda x: assign_group(x, cutOffPoint))
    #     regroup = calc_chi(df, col, target)[0]
    #     chisqlist = calc_chi(df, col, target)[1]
    #     ascecolList = regroup.sort_values(['badrate'], ascending=True)[col].tolist()
    #     desccolList = regroup.sort_values(['badrate'], ascending=False)[col].tolist()
    #     colcolList = regroup.sort_values([col])[col].tolist()
    #     while ascecolList != colcolList and desccolList != colcolList:
    #         minchiIndex = chisqlist.index(min(chisqlist))
    #         if minchiIndex == 0:
    #             cutOffPoint.remove(cutOffPoint[minchiIndex])
    #         elif minchiIndex == len(chisqlist):
    #             cutOffPoint.remove(cutOffPoint[minchiIndex - 1])
    #         else:
    #             if chisqlist[minchiIndex - 1] < chisqlist[minchiIndex]:
    #                 cutOffPoint.remove(cutOffPoint[minchiIndex - 1])
    #             else:
    #                 cutOffPoint.remove(cutOffPoint[minchiIndex])
    #         if cutOffPoint == []:
    #             return cutOffPoint
    #         df[col] = df[col].apply(lambda x: assign_group(x, cutOffPoint))
    #         regroup = calc_chi(df, col, target)[0]
    #         chisqlist = calc_chi(df, col, target)[1]
    #         ascecolList = regroup.sort_values(['badrate'], ascending=True)[col].tolist()
    #         desccolList = regroup.sort_values(['badrate'], ascending=False)[col].tolist()
    #         colcolList = regroup.sort_values([col])[col].tolist()
    #     return cutOffPoint



if __name__ == '__main__':
    # from woe import WOETools
    # dfxx = pd.read_csv("/Users/clay/Code/rocket/data/hl_test_clean.csv")
    # Binning.binning_manual(dfxx, "fpd", "hl_phone_silent_frequentcy",[0.05, 0.1, 0.15])
    # WOETools.insert_woe(dfxx, "hl_phone_silent_frequentcy", "fpd", [0.05, 0.1, 0.15])
    # df = pd.read_csv("/Users/clay/Code/scorecard/learn/qhxlj.csv")
    #
    # # 等距分箱
    # eq_bin = Binning.binning_equidistant(df, "package_fee_0", 4)
    # bif = BinningInfo(df, "package_fee_0", "y", eq_bin)
    # print(bif.bininfo())
    # print(bif.iv())
    # print(bif.bump())



    # 卡方分箱有问题
    # from outlib.lq.ZX_FeatureEngineering import chi_bin
    # ss = chi_bin(df, "package_fee_0", "y", 3)
    # print(ss)
    from outlib.ju.bin import Bins
    # ss = Binning.chimerge_XXXX(df, "package_fee_0", "y", max_intervals=10)
    # Binning.binning_manual(df, "package_fee_0", "y", ss)

    # ch_bin = Binning.binning_chi(df, "package_fee_0", "y", 5)
    # print(ch_bin)

    # # 等频分箱
    # freq_bin = Binning.binning_frequency(df, "package_fee_0", 5)
    # Binning.binning_manual(df, "y", "package_fee_0", freq_bin)
    #
    # # 决策树分箱
    # tree_bin = Binning.binning_tree(df, "package_fee_0", "y")
    # Binning.binning_manual(df, "y", "package_fee_0", tree_bin)

    aa = BinningTools.cut_str(["a", "b", "b", "a", "c", "c", "d", "E", None, "q"], ["a", ["b", "c"], "d"])
    print(aa)
