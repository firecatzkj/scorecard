from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import os
import json
import warnings

warnings.filterwarnings('ignore')


class Widget:
    '''
    The widget prepared for other class
    v1.0,copyright by AntiMage-Janhonho
    '''

    def __init__(self):
        pass

    @staticmethod
    def binary_check(data):
        '''
        Check the type of data(y) is 0-1 or not
        :param data: list_like data
        :return: True or False
        '''
        data = pd.Series(data).copy()

        return data.isin([0, 1])

    @staticmethod
    def nan_normalize(x):
        '''
        Normalize the so many type of nan
        :param x: list_like raw data
        :return: the normalized data
        '''

        x = pd.Series(x).copy()
        x = x.replace([' ', '', 'null', 'Null', 'NULL', 'nan',
                       'np.nan', 'Nan', 'NaN', 'NAN'], np.nan)

        return x

    @staticmethod
    def bin_type_guess(data):
        '''
        Guess the bin_type
        :param data: DataFrame
        :return: type_dict
        '''
        data = pd.DataFrame(data).copy()
        result_dict = {}
        for i in data.columns:
            try:
                data[i].apply(float)
            except Exception:
                result_dict[i] = 'M'
        return result_dict

    @staticmethod
    def drop_feature_vif(input_df, threld=10):
        '''
        Use variance_inflation to drop feature of the input_df
        :param input_df: DataFrame, input data
        :param threld: float, the threld of the variance inflation
        :return: list_like,the col left
        '''

        df = input_df.copy()
        all_cols = list(df.columns)
        max_iternum = len(all_cols)
        for i in np.arange(0, max_iternum):
            vif = pd.Series([variance_inflation_factor(df[all_cols].values, col_id)
                             for col_id in range(df[all_cols].shape[1])], index=all_cols)
            if vif.max() > threld:
                all_cols.remove(vif.idxmax(axis=1))
            else:
                break

        return all_cols

    @staticmethod
    def get_jump_point(input_array):
        '''
        Calculate the num of jump point & predict the direction of the jump point
        :param input_s: list_like,PD direction
        :return: tuple,the num of jump point & the predicted direction
        '''

        array = pd.Series(input_array).copy()
        negative_diff = array.diff().apply(np.sign)
        negative_diff = negative_diff.iloc[1:-1]
        negative_diff = negative_diff.replace(0, np.nan)
        negative_diff = negative_diff.fillna(method='ffill')

        positive_diff = array.diff(-1).apply(np.sign)
        positive_diff = positive_diff.iloc[1:-1]
        positive_diff = positive_diff.replace(0, np.nan)
        positive_diff = positive_diff.fillna(method='bfill')
        result = negative_diff * positive_diff
        jump_num = len(result[result > 0])

        direction = np.sign(array.diff().apply(np.sign).sum())

        return jump_num, direction


class DataPreprocessing(Widget):
    '''
    General preprocess for the data
    v1.0,copyright by AntiMage-Janhonho
    '''

    def __init__(self):
        pass




class Bins(Widget):
    '''
    Bin Class which can be used to generate bins of the original variable
    v1.0,copyright by AntiMage-Janhonho
    The work further to do:
    1.Binning which limit each bin's shape（monotonous or U-shape)
    2.Better interactive method
    3.GBDT-style or other strange style binning
    4.Bin size is more even
    '''

    def __init__(
            self,
            method=3,
            init_num=20,
            end_num=5,
            plimit=5,
            nlimit=5,
            threshold_value=np.nan,
            ftype=None):
        '''
        :param method: int,in fact the category,1:Entropy 2:Gini 3:Chi-square 4:Info-value
        :param init_num: int, the init num of bin
        :param end_num: int, the end num of bin
        :param plimit: int, the least positive sample num
        :param nlimit: int, the least negative sample num
        '''

        self.method = method
        self.init_num = init_num
        self.end_num = end_num
        self.plimit = plimit
        self.nlimit = nlimit
        self.threshold_value=threshold_value
        self.__bin_stat = pd.DataFrame()
        self.__bin_interval = []
        self.__bin_map = {}
        self.__ftype = ftype

        if method == 1:
            self.split_func = Bins.__Entropy
        elif method == 2:
            self.split_func = Bins.__Gini
        elif method == 3:
            self.split_func = Bins.__Chi_square
        elif method == 4:
            self.split_func = Bins.__Info_value
        else:
            raise NotImplementedError

    @staticmethod
    def __Gini(bin_df, split):
        '''
        Calucate the Gini Gain
        :param bin_df: DataFrame,bin_df
        :param split: string, which columns used to group
        :return: float,the Gini-Gain
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_total = df['total'].sum()
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()

        group = df.groupby(split)
        CG = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            part_sum_total = part_sum_0 + part_sum_1
            p_0 = part_sum_0 / part_sum_total
            p_1 = part_sum_1 / part_sum_total
            m_0 = p_0 * p_0
            m_1 = p_1 * p_1
            CG = CG + part_sum_total * (1 - m_0 - m_1) / df_sum_total

        init_p_0 = df_sum_0 / df_sum_total
        init_p_1 = df_sum_1 / df_sum_total
        init_G = 1 - init_p_0 * init_p_0 - init_p_1 * init_p_1

        delta_G = 1 - CG / init_G

        return delta_G

    @staticmethod
    def __Entropy(bin_df, split):
        '''
        Calucate the Entropy Gain
        :param bin_df: DataFrame bin_df
        :param split: string, which columns used to group
        :return: float,the Entropy-Gain
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_total = df['total'].sum()
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()

        group = df.groupby(split)
        CE = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            part_sum_total = part_sum_0 + part_sum_1
            p_0 = part_sum_0 / part_sum_total
            p_1 = part_sum_1 / part_sum_total
            m_0 = -p_0 * np.log2(p_0)
            m_1 = -p_1 * np.log2(p_1)
            CE = CE + part_sum_total * (m_0 + m_1) / df_sum_total

        init_p_0 = df_sum_0 / df_sum_total
        init_p_1 = df_sum_1 / df_sum_total
        init_E = -((init_p_0) * np.log2(init_p_0) +
                   (init_p_1) * np.log2(init_p_1))
        # the delta_E is a relative value
        Gain_E = 1 - CE / init_E

        return Gain_E

    @staticmethod
    def __Chi_square(bin_df, split):
        '''
        Calucate the Chi-square value, in fact from sklearn.feature_selection import chi2 can solve the problem
        :param bin_df: DataFrame,bin_df
        :param split: string, which columns used to group
        :return: float,the Chi_square value
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()
        df_sum_total = df['total'].sum()
        group = df.groupby(split)
        CS = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            part_sum_total = part_sum_0 + part_sum_1
            p_0 = part_sum_total * df_sum_0 / df_sum_total
            p_1 = part_sum_total * df_sum_1 / df_sum_total
            CS = CS + (part_sum_0 - p_0) ** 2 / p_0 + \
                (part_sum_1 - p_1) ** 2 / p_1

        return CS

    @staticmethod
    def __Info_value(bin_df, split):
        '''
        Calculate the IV
        :param bin_df: DataFrame bin_df
        :param split: string, which columns used to group
        :return: float,the Info_Value
        '''

        df = bin_df.copy()
        df['total'] = df[0.0] + df[1.0]
        df_sum_0 = df[0.0].sum()
        df_sum_1 = df[1.0].sum()
        group = df.groupby(split)
        IV = 0
        for name, part in group:
            part_sum_0 = part[0.0].sum()
            part_sum_1 = part[1.0].sum()
            p_0 = part_sum_0 / df_sum_0
            p_1 = part_sum_1 / df_sum_1
            IV = IV + (p_0 - p_1) * (np.log(p_0 / p_1))

        return IV

    @staticmethod
    def __intervals_to_list(intervals):
        '''
        Tool function used to return the interval edge
        :param intervals: list which the element is the interval
        :return: list,the bin edge
        '''

        intervals = pd.Series(intervals).copy()
        intervals = intervals.replace(
            [' ', '', 'null', 'Null', 'nan', 'np.nan', 'Nan', 'NaN', 'NAN', 'NULL'], np.nan)
        all_list = []
        for i in intervals.dropna():
            all_list.append(i.left)
            all_list.append(i.right)

        all_list.append(np.inf)
        all_list.append(-np.inf)

        bin_interval = sorted(set(all_list))

        return bin_interval

    def __C_binary_split(self, bin_df):
        '''
        Binary split the whole part
        :param bin_df:
        :return: the split result of one part
        '''

        bin_df = bin_df.copy()
        maxv = self.threshold_value
        result = bin_df
        for i in range(0, len(bin_df.index) - 1):
            tmp = bin_df.copy()
            left = bin_df.ix[i, 'nbins'].left
            split_num = bin_df.index[i].right
            right = bin_df.ix[i, 'nbins'].right
            tmp.ix[0:i + 1, 'nbins'] = pd.Interval(left, split_num)
            tmp.ix[i + 1:, 'nbins'] = pd.Interval(split_num, right)
            value = self.split_func(tmp, 'nbins')
            if pd.isnull(maxv) or value > maxv:
                maxv = value
                result = tmp

        return result

    def __C_split(self, bin_df):
        '''
        Split choose one bin to split
        :param bin_df: the bin_df to split
        :return: one-time split result
        '''

        bin_df = bin_df.copy()
        group = bin_df.groupby('nbins')
        result = bin_df.copy()
        maxv = self.threshold_value

        tmp_r = []

        for name, part in group:
            tmp = bin_df.copy()
            if len(part.index) > 1:
                part_bin_df = self.__C_binary_split(part)
                tmp.ix[tmp.index.get_loc(part_bin_df.index[0]):tmp.index.get_loc(
                    part_bin_df.index[-1]) + 1, 'nbins'] = part_bin_df['nbins']
                value = self.split_func(tmp, 'nbins')
                tmp_r.append(value)
                if pd.isnull(maxv) or value > maxv:
                    maxv = value
                    result = tmp

        return result

    def __C_bin_reduce(self, bin_df):
        '''
        Get the bin_dict to reduce the bin
        :param bin_df: dataframe which is raw
        :return the cut which has been combined simply
        '''

        bin_df = bin_df.copy()
        num = 0
        for i in range(1, len(bin_df.index) + 1):
            psum = bin_df.ix[num:i, 0].sum()
            nsum = bin_df.ix[num:i, 1].sum()
            if psum >= self.plimit and nsum >= self.nlimit:
                bin_df.ix[num:i, 'nbin'] = pd.Interval(
                    bin_df.index[num].left, bin_df.index[i - 1].right)
                num = i
            if i == len(bin_df.index) and num < i:
                change_bin = bin_df.ix[num - 1, 'nbin']
                if pd.isnull(change_bin):
                    bin_df.ix[num:i, 'nbin'] = pd.Interval(
                        bin_df.index[num].left, bin_df.index[i - 1].right)
                else:
                    bin_df.ix[num:i, 'nbin'] = pd.Interval(
                        change_bin.left, bin_df.index[i - 1].right)
                    bin_df.ix[bin_df['nbin'] == change_bin, 'nbin'] = pd.Interval(
                        change_bin.left, bin_df.index[i - 1].right)

        bin_dict = bin_df['nbin'].to_dict()

        return bin_dict

    def __C_get_cut_result(self, cut, y):
        '''
        Combine the bin
        :param cut: interval,the cut to combine
        :param y: the label
        :return: the combined cut
        '''

        cut = cut.copy()
        y = pd.Series(y).copy()
        bin_df = pd.crosstab(index=cut, columns=y)
        left = bin_df.sort_index().index[0].left
        right = bin_df.sort_index().index[-1].right
        bin_df['nbins'] = pd.Interval(left, right)

        real_enum = min(self.end_num, len(bin_df.index))
        # print(real_enum, "}}}}}}}}}}}}}}}}}}}", self.end_num, len(bin_df.index))
        nbins = 1
        while (nbins < real_enum):
            bin_df = self.__C_split(bin_df)
            nbins += 1

        bin_df = bin_df.groupby('nbins').sum().sort_index()

        return bin_df

    @staticmethod
    def __MD_preprocessing(ftype, x, y, **kwargs):
        '''
        Transform the discrete data into continuous data
        :param x: list_like, raw data
        :param y: list_like, label data
        :return:list_like, preprocessing data
        '''

        x = pd.Series(x).copy()
        x = x.apply(str)
        x.index = pd.Series(x.index).apply(str)
        x = Bins.nan_normalize(x)
        # here nan is not included

        if 'seq' in kwargs.keys():
            tmp = x.value_counts()
            seq = kwargs['seq']
            seq = pd.Series(seq).copy()
            seq = Bins.nan_normalize(seq)
            # because tmp has no nan so drop it in the seq
            seq = seq.dropna()
            tmp = tmp.reindex(seq)
        elif ftype == 'M':
            tmp = x.value_counts()
            tmp = tmp.sort_index()
        elif ftype == 'D':
            tmp = pd.crosstab(x, y)
            tmp['brate'] = tmp[1] / (tmp[0] + tmp[1])
            tmp = tmp.sort_values(by='brate')
        else:
            raise ValueError('Without seq,Only M or D is permitted')

        map_dict = pd.Series(range(0, len(tmp.index)),
                             index=tmp.index).to_dict()
        xnum = x.apply(lambda x: map_dict.get(x, np.nan))

        return xnum, map_dict

    @staticmethod
    def __MD_reprocessing(bin_df, map_dict):
        '''
        Transform the interval into discrete set
        :param bin_df: bin_stat
        :param map_dict: dict like
        :return: the reprocess result
        '''

        bin_df = bin_df.copy()

        tmp = pd.Series(map_dict)
        tmp.name = 'num'
        tmp.index.name = 'OBin'
        tmp = tmp.reset_index()
        mapping = tmp.set_index('num')['OBin'].to_dict()

        result_dict = {}
        for j in range(0, len(bin_df.index)):
            result_dict[bin_df.index[j]] = []

        mkl = sorted(mapping.keys())
        for i in mkl:
            for j in range(0, len(bin_df.index)):
                if i in bin_df.index[j]:
                    result_dict[bin_df.index[j]].append(mapping[i])

        bin_df = bin_df.rename(index=result_dict)

        return bin_df

    def __generate_bin_stat_without_interval_C(self, x, y):
        '''
        Generate the bin_stat of continuous data accroding to the x,y
        :param x:list_like,the continuous x
        :param y:list_like, the label
        :return:DataFrame, bin_stat
        '''
        x = pd.Series(x).copy()
        x = Bins.nan_normalize(x)
        x = x.apply(float)
        y = pd.Series(y).copy()
        y = Bins.nan_normalize(y)

        try:
            init_cut = pd.qcut(x, self.init_num, duplicates='drop')
            retbin = pd.Series(init_cut.values.categories).sort_values()
            retbin.iloc[0] = pd.Interval(-np.inf, retbin.iloc[0].right)
            retbin.iloc[-1] = pd.Interval(retbin.iloc[-1].left, np.inf)
            init_cut = pd.cut(x, pd.IntervalIndex(retbin))
            init_cut = init_cut.astype(pd.Interval)
        except IndexError:
            init_cut = x.copy()
            init_cut[pd.notnull(init_cut)] = pd.Interval(-np.inf, np.inf)
            retbin = pd.Series(pd.Interval(-np.inf, np.inf))

        bin_df = pd.crosstab(index=init_cut, columns=y)
        bin_df = bin_df.reindex(retbin)
        bin_df = bin_df.sort_index()
        bin_df = bin_df.fillna(0.0)

        bin_df['nbin'] = np.nan

        bin_dict = self.__C_bin_reduce(bin_df)
        combine_cut = init_cut.map(bin_dict)
        bin_stat = self.__C_get_cut_result(combine_cut, y)

        return bin_stat

    def __generate_bin_stat_without_interval_MD(self, x, y, **kwargs):
        '''
        Generate the bin_stat of MD data accroding to the x,y
        :param x:list_like,the continuous x
        :param y:list_like, the label
        :return:DataFrame, bin_stat
        '''

        x, map_dict = Bins.__MD_preprocessing(self.__ftype, x, y, **kwargs)
        bin_stat = self.__generate_bin_stat_without_interval_C(x, y)
        bin_stat = Bins.__MD_reprocessing(bin_stat, map_dict)

        return bin_stat

    def __generate_bin_stat_without_interval(self, x, y, **kwargs):
        '''
        Generate bin_stat with interval
        :param x: list_like ,the data
        :param y: list_like,the label
        :param interval: list_like,the interval
        :return: DataFrame, bin_stat
        '''

        if self.__ftype == 'C':
            bin_stat = self.__generate_bin_stat_without_interval_C(x, y)
        elif self.__ftype == 'M' or self.__ftype == 'D':
            bin_stat = self.__generate_bin_stat_without_interval_MD(
                x, y, **kwargs)
        else:
            raise NotImplementedError

        return bin_stat

    def __generate_bin_stat_with_interval_C(self, x, y, interval):
        '''
        Generate the bin_stat with continuous data which has interval
        :param x:list_like, raw data
        :param y:list_like, the label
        :param interval: list_like the interval
        :return:DataFrame, bin_stat
        '''

        x = pd.Series(x).copy()
        x = Bins.nan_normalize(x)
        x = x.apply(float)
        y = pd.Series(y).copy()
        y = Bins.nan_normalize(y)
        interval = pd.Series(interval).copy()
        interval = Bins.nan_normalize(interval)
        interval = interval.dropna()
        interval = list(set(interval).union([-np.inf, np.inf]))
        interval = pd.Series(interval)
        interval = interval.sort_values()
        interval.index = range(0, len(interval.index))
        interval.index = pd.Series(interval.index).apply(str)

        interval_list = []
        for i in range(0, len(interval.index) - 1):
            interval_list.append(pd.Interval(
                interval.ix[i], interval.ix[i + 1]))

        init_cut = pd.cut(x, pd.IntervalIndex(interval_list))
        retbin = pd.Series(init_cut.values.categories).sort_values()

        init_cut = pd.cut(x, pd.IntervalIndex(retbin))
        init_cut = init_cut.astype(pd.Interval)

        bin_df = pd.crosstab(index=init_cut, columns=y)
        bin_df = bin_df.reindex(retbin)
        bin_df = bin_df.sort_index()
        bin_df = bin_df.fillna(0.0)

        bin_stat = bin_df

        return bin_stat

    def __generate_bin_stat_with_interval_MD(self, x, y, interval, **kwargs):
        '''
        Generate the bin_stat with Distinct(M,D) data which has interval
        :param x:list_like, raw data
        :param y:list_like, the label
        :param interval: list_like the interval
        :return:DataFrame, bin_stat
        '''

        x = pd.Series(x).copy()
        x = x.apply(str)
        x.index = pd.Series(x.index).apply(str)
        x = Bins.nan_normalize(x)
        x = x.fillna('NaN')
        y = pd.Series(y).copy()
        y = Bins.nan_normalize(y)
        interval = pd.Series(interval).copy()

        if 'seq' in kwargs.keys():
            seq = kwargs['seq']
            seq = Bins.nan_normalize(seq)
            seq = seq.dropna()
            seq_df = pd.DataFrame({'seq': seq, 'num': range(0, len(seq))})
            num_len = int(np.log10(len(seq))) + 2
            seq_df['num'] = seq_df['num'].apply(
                lambda x: '{:0>{}}'.format(x, num_len))
            seq_dict = seq_df.set_index('seq')['num'].to_dict()
            re_seq_dict = seq_df.set_index('num')['seq'].to_dict()
        else:
            seq_dict = None
            re_seq_dict = None

        if seq_dict is not None:
            x_c = x.apply(lambda x: seq_dict.get(x, 'NaN'))
            interval_c = []
            for i in range(0, len(interval)):
                tmp = interval[i]
                if isinstance(tmp, tuple) or isinstance(tmp, list):
                    tmp = pd.Series(tmp).copy()
                    tmp = tmp.apply(str)
                    tmp = Bins.nan_normalize(tmp)
                    tmp = tmp.apply(lambda x: seq_dict.get(x, np.nan))
                    interval_c.append(list(tmp))
                else:
                    tmp_s = pd.Series([tmp]).apply(str)
                    tmp_s = Bins.nan_normalize(tmp_s)
                    tmp_s = tmp_s.fillna('NaN')
                    tmp = tmp_s.iloc[0]
                    tmp = str(tmp)
                    interval_c.append(seq_dict.get(tmp, np.nan))

            interval_c = pd.Series(interval_c)

        else:
            x_c = x.copy()
            interval_c = interval.copy()

        interval_map = {}
        all_key = []

        for i in range(0, len(interval_c)):
            tmp = interval_c[i]
            if isinstance(tmp, tuple) or isinstance(tmp, list):
                tmp = pd.Series(tmp).copy()
                tmp = tmp.apply(str)
                tmp = Bins.nan_normalize(tmp)
                tmp = tmp.sort_values()
                tmp = tmp.fillna('NaN')
                name = []
                for j in range(0, len(tmp)):
                    name.append(tmp.iloc[j])
                    all_key.append(tmp.iloc[j])
                for j in range(0, len(tmp)):
                    interval_map[tmp.iloc[j]] = name
            else:
                tmp_s = pd.Series([tmp]).apply(str)
                tmp_s = Bins.nan_normalize(tmp_s)
                tmp_s = tmp_s.fillna('NaN')
                tmp = tmp_s.iloc[0]
                tmp = str(tmp)
                interval_map[tmp] = [tmp]
                all_key.append(tmp)

        all_x = x_c.unique()

        for i in all_x:
            if i not in all_key:
                interval_map[i] = [i]

        df = pd.DataFrame({'OBin': pd.Series(interval_map)})
        df['strOBin'] = df['OBin'].apply(str)
        str_dict = df['strOBin'].to_dict()

        x_c = x_c.apply(lambda x: str_dict[x])
        bin_stat = pd.crosstab(x_c, y)

        tmp = df[['OBin', 'strOBin']].copy()
        tmp = tmp.drop_duplicates(subset=['strOBin'])
        tmp = tmp.set_index('strOBin')
        re_dict = tmp['OBin'].to_dict()
        if seq_dict is None:
            bin_stat = bin_stat.rename(index=re_dict)
        else:
            re_dict2 = {}
            for i in re_dict.keys():
                tmp = list(
                    pd.Series(
                        re_dict[i]).apply(
                        lambda x: re_seq_dict.get(
                            x, 'NaN')))
                re_dict2[i] = tmp
            bin_stat = bin_stat.rename(index=re_dict2)

        return bin_stat

    def __generate_bin_stat_with_interval(self, x, y, interval, **kwargs):
        '''
        Generate bin_stat with interval
        :param x: list_like ,the data
        :param y: list_like,the label
        :param interval: list_like,the interval
        :return: DataFrame, bin_stat
        '''

        if self.__ftype == 'C':
            bin_stat = self.__generate_bin_stat_with_interval_C(x, y, interval)
        elif self.__ftype == 'M' or self.__ftype == 'D':
            bin_stat = self.__generate_bin_stat_with_interval_MD(
                x, y, interval, **kwargs)
        else:
            raise NotImplementedError

        return bin_stat

    def __generate_bin_stat(self, x, y, interval, **kwargs):
        '''
        Get the bin_stat of the continuous variable
        :param x: list_like,the feature
        :param y: list_like,the label
        :param input_interval: list_like, user-defined interval
        '''

        x = pd.Series(x).copy()
        y = pd.Series(y).copy()
        y = y.reindex(x.index)

        x = Bins.nan_normalize(x)
        x.index = range(0, len(x.index))
        x.index = pd.Series(x.index).apply(str)

        y = Bins.nan_normalize(y)
        y.index = range(0, len(y.index))
        y.index = pd.Series(y.index).apply(str)

        y = y.reindex(x.index)

        if pd.isnull(y).any():
            raise ValueError("y has NaN")

        if interval is None:
            bin_stat = self.__generate_bin_stat_without_interval(
                x, y, **kwargs)
        else:
            bin_stat = self.__generate_bin_stat_with_interval(
                x, y, interval, **kwargs)

        bin_stat.index.name = 'Interval'
        bin_stat = bin_stat.reset_index()

        all_key = []
        for i in range(0, len(bin_stat.index)):
            if isinstance(bin_stat['Interval'].iloc[i], list):
                all_key = all_key + bin_stat['Interval'].iloc[i]
            else:
                all_key.append(bin_stat['Interval'].iloc[i])

        # deal with NaN if NaN is not in the interval
        if 'NaN' not in all_key:
            y1_nan = y.ix[pd.isnull(x)].sum()
            y0_nan = y.ix[pd.isnull(x)].shape[0] - y1_nan
            bin_stat = bin_stat.append(pd.DataFrame(
                {'Interval': ['NaN'], 0: [y0_nan], 1: [y1_nan]}, index=[bin_stat.shape[0]]))

        if 0 not in bin_stat.columns:
            bin_stat[0] = np.nan
        if 1 not in bin_stat.columns:
            bin_stat[1] = np.nan

        # in very seldom condition, only one bin is returned,so take care of it
        bin_stat['type'] = self.__ftype

        bin_stat['Bin'] = range(1, bin_stat.shape[0] + 1)
        num_len = int(np.log10(len(bin_stat.index))) + 2
        bin_stat['Bin'] = bin_stat['Bin'].apply(
            lambda x: 'B{:0>{}}'.format(x, num_len))

        if self.__ftype == 'C':
            bin_stat['lower_json'] = np.nan
            bin_stat['upper_json'] = np.nan
            for j in range(0, len(bin_stat.index)):
                if bin_stat.ix[j, 'Interval'] == 'NaN':
                    bin_stat.ix[j, 'lower_json'] = 'NaN'
                    bin_stat.ix[j, 'upper_json'] = 'NaN'
                else:
                    bin_stat.ix[j, 'lower_json'] = json.dumps(
                        bin_stat.ix[j, 'Interval'].left)
                    bin_stat.ix[j, 'upper_json'] = json.dumps(
                        bin_stat.ix[j, 'Interval'].right)
        else:
            bin_stat['lower_json'] = bin_stat['Interval'].apply(
                lambda x: json.dumps(x))
            bin_stat['upper_json'] = bin_stat['Interval'].apply(
                lambda x: json.dumps(x))

        self.__bin_stat = bin_stat.copy()

    @staticmethod
    def __stat_to_interval(bin_stat, ftype):
        '''
        Tool func generate bin_interval from bin_stat
        :param bin_stat: DataFrame, bin_stat
        :param ftype: str,data type
        :return: list,bin_interval
        '''

        bin_stat = bin_stat.copy()
        if ftype == 'C':
            bin_interval = Bins.__intervals_to_list(bin_stat['Interval'])
        elif ftype == 'M' or ftype == 'D':
            bin_interval = list(bin_stat['Interval'])
        else:
            raise NotImplementedError

        return bin_interval

    def __generate_bin_interval(self):

        bin_stat = self.__bin_stat.copy()
        bin_interval = Bins.__stat_to_interval(bin_stat, self.__ftype)
        self.__bin_interval = bin_interval

    @staticmethod
    def __stat_to_map(bin_stat, ftype):
        '''
        Tool func generate bin_map from bin_stat
        :param bin_stat: DataFrame, bin_stat
        :param ftype: str,data type
        :return: dict,bin_map
        '''

        bin_stat = bin_stat.copy()
        if ftype == 'C':
            tmp = bin_stat.set_index('Interval')
            bin_map = tmp['Bin'].to_dict()
        elif ftype == 'M' or ftype == 'D':
            BI = bin_stat[['Interval', 'Bin']]
            bin_map = {}
            for i in range(0, len(BI.index)):
                tmp = BI['Interval'].iloc[i]
                if isinstance(tmp, tuple) or isinstance(tmp, list):
                    for j in tmp:
                        bin_map[j] = BI['Bin'].iloc[i]
                else:
                    bin_map[tmp] = BI['Bin'].iloc[i]
        else:
            raise NotImplementedError

        return bin_map

    def __generate_bin_map(self):

        bin_stat = self.__bin_stat.copy()
        bin_map = Bins.__stat_to_map(bin_stat, self.__ftype)
        self.__bin_map = bin_map

    def generate_bin_smi(self, x, y, interval=None, ftype=None, **kwargs):
        '''
        Get the bin_interval&bin_map which can be used
        :param x: list_like,the feature
        :param y: list_like,the label
        :param ftype: str, C means continuous data,others means discrete data
        :param interval: list_like, user-defined interval
        :param **kwargs: other parameter
        '''

        if ftype is None and self.__ftype is None:
            raise ValueError('Ftype is needed')

        if ftype is not None:
            self.__ftype = ftype

        if Bins.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        self.__generate_bin_stat(x, y, interval, **kwargs)
        self.__generate_bin_map()
        self.__generate_bin_interval()

    def get_bin_info(self):
        '''
        Return the bin_stat,bin_map& bin_interval to the user
        :return: tuple,bin_stat, bin_interval, bin_map
        '''

        bin_stat = self.__bin_stat.copy()
        bin_stat[[0, 1]] = bin_stat[[0, 1]].fillna(0.0)
        bin_stat['total'] = bin_stat[1] + bin_stat[0]
        bin_stat['PD'] = bin_stat[1] / bin_stat['total']
        bin_stat['1_prop'] = bin_stat[1] / bin_stat[1].sum()
        bin_stat['0_prop'] = bin_stat[0] / bin_stat[0].sum()
        bin_stat['total_prop'] = bin_stat['total'] / bin_stat['total'].sum()
        jump_num, direction = Bins.get_jump_point(bin_stat['PD'].iloc[:-1])
        bin_stat['jn'] = jump_num
        bin_stat['direction'] = direction

        bin_stat.index = pd.Series(bin_stat.index).apply(str)
        bin_interval = copy.deepcopy(self.__bin_interval)
        bin_map = copy.deepcopy(self.__bin_map)

        return bin_stat, bin_interval, bin_map

    def value_to_bin(self, x):
        '''
        From init value to bin_num
        :param x: Series,init value
        :return Series,the replaced value
        '''

        x = pd.Series(x).copy()
        x = Bins.nan_normalize(x)
        if self.__ftype == 'C':
            x = x.apply(float)
            tmp = self.__bin_interval
            interval_list = []
            for i in range(0, len(tmp) - 1):
                interval_list.append(pd.Interval(tmp[i], tmp[i + 1]))
            interval_x = pd.cut(
                x, pd.IntervalIndex(interval_list)).astype(
                pd.Interval)
            interval_x = interval_x.fillna('NaN')
            result = interval_x.apply(lambda x: self.__bin_map[x])
        elif self.__ftype == 'M' or self.__ftype == 'D':
            x = x.apply(str)
            x = Bins.nan_normalize(x)
            x = x.fillna('NaN')
            num_len = len(list(self.__bin_map.values())[0]) - 1
            result = x.apply(
                lambda x: self.__bin_map.get(
                    x, 'B{:0>{}}'.format(
                        0, num_len)))
        else:
            raise NotImplementedError

        return result

    @staticmethod
    def bin_replace(x, interval=None, ftype='C', y=None):
        '''
        Replace the orignal data with the bin
        :param x: list_like, original data
        :param interval: list_like
        :param ftype: str,C or D
        :param y:list_like,the corresponding y
        :return: tuple,result, bin_stat, bin_map
        '''

        x = pd.Series(x).copy()
        if y is None:
            y = pd.Series(0, index=x.index)
        else:
            y = pd.Series(y).copy()

        if Bins.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        tmp = Bins()
        if ftype == 'C':
            if interval is None:
                raise ValueError('No Interval Error!')
            tmp.generate_bin_smi(x, y, interval, ftype='C')
        elif ftype == 'M':
            if interval is None:
                interval = [np.nan]
            tmp.generate_bin_smi(x, y, interval, ftype='M')
        elif ftype == 'D':
            if interval is None:
                interval = [np.nan]
            tmp.generate_bin_smi(x, y, interval, ftype='D')
        else:
            raise ValueError('Ftype Error!')

        bin_stat, bin_interval, bin_map = tmp.get_bin_info()
        result = tmp.value_to_bin(x)

        return result, bin_stat, bin_map

    @staticmethod
    def __woe_iv(x, y, all_bin):
        '''
        Caculate the woe
        :param x: list_like,the feature
        :param y: list_like,the label
        :param all_bin: list_like, all bin
        :return: tuple,iv_contribution, woe, xytable
        '''
        x = pd.Series(x).copy()
        y = pd.Series(y).copy()

        if len(y) > 0:
            yvc = y.value_counts()
            yvc = yvc.reindex([0, 1]).fillna(0.0)

            total_0 = yvc[0]
            total_1 = yvc[1]
            xytable = pd.crosstab(x, y)
            xytable = xytable.reindex(columns=[0, 1]).fillna(0.0)

            xytable.index = pd.Series(xytable.index).apply(str)
            if all_bin is not None:
                xytable = xytable.reindex(xytable.index.union(all_bin))
                xytable = xytable.fillna(0.0)

            relative_0 = xytable[0] / total_0
            relative_1 = xytable[1] / total_1
            woe = (relative_1 / relative_0).apply(np.log)
            iv_contribution = (relative_1 - relative_0) * woe
        else:
            iv_contribution = pd.Series()
            woe = pd.Series()
            xytable = pd.DataFrame()

        return iv_contribution, woe, xytable

    @staticmethod
    def woe_iv(x, y=None, all_bin=None, report=None,modified_flag = 0,split_bin=[]):
        '''
        Caculate the woe
        :param x: list_like,the feature
        :param y: list_like,the label
        :param all_bin: list_like, all bin
        :param report: dataframe, the woe_iv report
        :param split_bin: list_like,often the NaN bin
        :return: tuple,result, report, error_flag
        '''

        x = pd.Series(x).copy()
        x = x.apply(str)

        all_bin = pd.Series(all_bin).copy()

        r_split_bin = list(pd.Series(split_bin))

        if report is None:
            if y is None:
                raise ValueError('No report Nor y!')
            else:
                if Bins.binary_check(y) is False:
                    raise ValueError('Value Error of y!')

                iv_series, woe, xytable = Bins.__woe_iv(x, y, all_bin)
                result = iv_series.replace([-np.inf, np.inf, np.nan], 0.0)
                iv_values = pd.Series(result.sum(), index=result.index)

                # here sometimes duplicates index may become a big problem!
                split_x = x[-x.isin(r_split_bin)].copy()
                split_y = y.reindex(split_x.index)
                split_iv_series, split_woe, split_xytable = Bins.__woe_iv(
                    split_x, split_y, all_bin[-all_bin.isin(r_split_bin)])
                split_result = split_iv_series.replace(
                    [-np.inf, np.inf, np.nan], 0.0)
                split_iv_values = pd.Series(
                    split_result.sum(), index=result.index)

                if len(r_split_bin) == 0:
                    binary_iv_values = np.nan
                else:
                    binary_x = x.replace(list(r_split_bin), 'binary_1').copy()
                    binary_x[binary_x != 'binary_1'] = 'binary_0'
                    binary_iv_series, binary_woe, binary_xytable = Bins.__woe_iv(
                        binary_x, y, ['binary_0', 'binary_1'])
                    binary_result = binary_iv_series.replace(
                        [-np.inf, np.inf, np.nan], 0.0)
                    binary_iv_values = pd.Series(
                        binary_result.sum(), index=result.index)

                if modified_flag == 1:
                    wmax = woe.replace([np.inf,-np.inf],np.nan).dropna().max()
                    wmin = woe.replace([np.inf,-np.inf],np.nan).dropna().min()
                    woe = woe.replace(np.inf,wmax)
                    woe = woe.replace(-np.inf,wmin)
                    woe = woe.fillna(wmax)
                    woe.iloc[-1] = wmax

                report = pd.DataFrame({'woe': woe,
                                       'ivc': iv_series,
                                       'iv': iv_values,
                                       'part_iv': split_iv_values,
                                       'binary_iv': binary_iv_values,
                                       '0': xytable[0],
                                       '1': xytable[1],
                                       'total': xytable[0] + xytable[1]})
                report.index.name = 'Bin'
                report = report.reset_index()
                woe_dict = woe.to_dict()
                result = x.apply(lambda x: woe_dict[x])
                error_flag = False
        else:
            report = report.copy()
            report = report.set_index('Bin')
            woe = report['woe'].copy()
            woe_dict = woe.to_dict()
            result = x.apply(lambda x: woe_dict.get(x, np.nan))
            error_flag = pd.isnull(result).any()

        return result, report, error_flag

    # @staticmethod
    def generate_raw(self, data, y, type_dict={}, MDP=False):
        '''
        Generate the report quickly
        :param data: DataFrame,the original data
        :param y: list_like,the label
        :param type_dict: dict_like,each type of the feature
        :param MDP: bool, whether to process the MD data
        :return: tuple, all_report, all_change_report, woe_df, bin_df, error_dict
        '''

        data = data.copy()

        all_report = pd.DataFrame()
        all_change_report = pd.DataFrame()

        tmp = self
        bin_df = data.copy()
        woe_df = data.copy()

        type_dict_predict = type_dict
        specified_col = list(type_dict.keys())
        left_type_dict_predict = self.bin_type_guess(
            data.drop(columns=specified_col, errors='ignore'))
        type_dict_predict.update(left_type_dict_predict)

        type_s = pd.Series(type_dict_predict).copy()
        type_s = type_s.reindex(data.columns)
        type_s = type_s.fillna('C')

        error_dict = {}

        for i in range(0, len(data.columns)):
            try:
                part = data[data.columns[i]].copy()
                if type_s.iloc[i] != 'C' and MDP == False:
                    interval = [np.nan]
                    tmp.generate_bin_smi(
                        part, y, interval=interval, ftype=type_s.iloc[i])
                else:
                    tmp.generate_bin_smi(part, y, ftype=type_s.iloc[i])

                bin_stat, bin_interval, bin_map = tmp.get_bin_info()
                bin_stat = bin_stat.drop(columns=[0, 1, 'total'])

                result = tmp.value_to_bin(part)

                split_bin = []
                for m in range(0, len(bin_stat.index)):
                    interval_tmp = bin_stat['Interval'].iloc[m]
                    if isinstance(interval_tmp, list):
                        for n in interval_tmp:
                            if 'NaN' in n:
                                split_bin.append(bin_stat['Bin'].iloc[m])
                                break
                    else:
                        split_bin = split_bin + \
                            list(bin_stat.ix[bin_stat['Interval'] == 'NaN', 'Bin'])

                split_bin = list(set(split_bin))

                woe_result, woeiv, error_flag = self.woe_iv(
                    result, y, all_bin=list(bin_stat['Bin']),modified_flag = 1,split_bin=split_bin)

                report = pd.merge(
                    bin_stat, woeiv, left_on='Bin', right_on='Bin')
                report['Feature'] = data.columns[i]

                woe_df[data.columns[i]] = woe_result
                bin_df[data.columns[i]] = result
                all_report = all_report.append(report)

                if bin_stat['type'].iloc[0] == 'C':
                    bin_interval = bin_interval[1:-1]

                change_report = pd.DataFrame({'Feature': [data.columns[i]], 'Interval': [
                    bin_interval], 'type': [type_s.iloc[i]]})
                change_report['Interval'] = change_report['Interval'].apply(
                    lambda x: json.dumps(x))
                all_change_report = all_change_report.append(
                    change_report, ignore_index=True)

            except Exception as err:
                error_dict[data.columns[i]] = err

        all_report = all_report.sort_values(
            by=['iv', 'Feature', 'Bin'], ascending=[False, True, True])

        col_list = list(all_report.columns)
        col_list.remove('Feature')
        all_report = all_report[['Feature'] + col_list]

        bin_df = bin_df.drop(columns=list(error_dict.keys()))
        woe_df = woe_df.drop(columns=list(error_dict.keys()))

        return all_report, all_change_report, woe_df, bin_df, error_dict

    # def generate_raw_mono(self, data: pd.DataFrame, y: list, type_dict, MDP=False, jn=0):
    #     """
    #     单调分箱
    #     :param data: 数据集
    #     :param y: y
    #     :param all_report:
    #     :param change_report:
    #     :return: 同generate_raw
    #     """
    #     all_report, change_report, woe_df, bin_df, false_dict = self.generate_raw(data, y, type_dict, MDP=MDP)
    #     var_monoed = list(all_report[all_report["jn"] == 0]["Feature"].unique())
    #     var_not_monoed = list(all_report[all_report["jn"] != 0]["Feature"].unique())
    #     change_report_mono_all = change_report[change_report["Feature"].isin(var_monoed)]
    #
    #     if len(var_not_monoed) == 0:
    #         return all_report, change_report, woe_df, bin_df, false_dict
    #     else:
    #         while True:
    #             self.end_num -= 1
    #             n_limit = self.end_num
    #             s_data = data[var_not_monoed]
    #             all_report, change_report, woe_df, bin_df, false_dict = self.generate_raw(s_data, y, type_dict, MDP=MDP)
    #             var_monoed = list(all_report[all_report["jn"] == 0]["Feature"].unique())
    #             var_not_monoed = list(all_report[all_report["jn"] != 0]["Feature"].unique())
    #             # change_report_no_mono = all_report[all_report["Feature"].isin(var_not_monoed)]
    #             change_report_mono = change_report[change_report["Feature"].isin(var_monoed)]
    #             change_report_mono_all = pd.concat([change_report_mono_all, change_report_mono], axis=0)
    #
    #             if len(var_not_monoed) == 0:
    #                 all_report_new = Bins.mannual_rebin(data, change_report_mono_all, y)
    #                 woe_df_new = Bins.whole_woe_replace(data, all_report_new)
    #                 return all_report_new, change_report_mono_all, woe_df_new, pd.DataFrame({}), pd.DataFrame({})
    #             if n_limit == 2:
    #                 if len(var_not_monoed) != 0:
    #                     # print(change_report_mono_all.columns)
    #                     # print(change_report_no_mono.columns)
    #                     change_report_no_mono = change_report[change_report["Feature"].isin(var_not_monoed)]
    #                     change_report_mono_all = pd.concat([change_report_mono_all, change_report_no_mono], axis=0)
    #                 # print(change_report_mono_all.to_csv("chg_mono.csv"))
    #                 all_report_new = Bins.mannual_rebin(data, change_report_mono_all, y)
    #                 woe_df_new = Bins.whole_woe_replace(data, all_report_new)
    #                 return all_report_new, change_report_mono_all, woe_df_new, pd.DataFrame({}), pd.DataFrame({})
    #

    def generate_raw_mono(self, data: pd.DataFrame, y: list, type_dict, MDP=False, jn=0):
        """
        单调分箱
        :param data: 数据集
        :param y: y
        :param all_report:
        :param change_report:
        :return: 同generate_raw
        """
        all_report, change_report, woe_df, bin_df, false_dict = self.generate_raw(data, y, type_dict, MDP=MDP)
        var_monoed = list(all_report[all_report["jn"] <= jn]["Feature"].unique())
        var_not_monoed = list(all_report[all_report["jn"] > jn]["Feature"].unique())
        change_report_mono_all = change_report[change_report["Feature"].isin(var_monoed)]

        if len(var_not_monoed) == 0:
            return all_report, change_report, woe_df, bin_df, false_dict
        else:
            while True:
                self.end_num -= 1
                n_limit = self.end_num
                s_data = data[var_not_monoed]
                all_report, change_report, woe_df, bin_df, false_dict = self.generate_raw(s_data, y, type_dict, MDP=MDP)
                var_monoed = list(all_report[all_report["jn"] <= jn]["Feature"].unique())
                var_not_monoed = list(all_report[all_report["jn"] > jn]["Feature"].unique())
                # change_report_no_mono = all_report[all_report["Feature"].isin(var_not_monoed)]
                change_report_mono = change_report[change_report["Feature"].isin(var_monoed)]
                change_report_mono_all = pd.concat([change_report_mono_all, change_report_mono], axis=0)

                if len(var_not_monoed) == 0:
                    all_report_new = Bins.mannual_rebin(data, change_report_mono_all, y)
                    woe_df_new = Bins.whole_woe_replace(data, all_report_new)
                    return all_report_new, change_report_mono_all, woe_df_new, pd.DataFrame({}), pd.DataFrame({})
                if n_limit == 2:
                    if len(var_not_monoed) != 0:
                        # print(change_report_mono_all.columns)
                        # print(change_report_no_mono.columns)
                        change_report_no_mono = change_report[change_report["Feature"].isin(var_not_monoed)]
                        change_report_mono_all = pd.concat([change_report_mono_all, change_report_no_mono], axis=0)
                    # print(change_report_mono_all.to_csv("chg_mono.csv"))
                    all_report_new = Bins.mannual_rebin(data, change_report_mono_all, y)
                    woe_df_new = Bins.whole_woe_replace(data, all_report_new)
                    return all_report_new, change_report_mono_all, woe_df_new, pd.DataFrame({}), pd.DataFrame({})



    @staticmethod
    def mannual_rebin(data, mreport, y=None):
        '''
        Quick method to solve multi-feature
        :param data: DataFrame,the original data
        :param mreport: DataFrame,mannual report
        :param y: the label
        :return: new bin report
        '''

        data = data.copy()
        mreport = mreport.copy()
        if isinstance(mreport['Interval'].iloc[0], str):
            mreport['Interval'] = mreport['Interval'].apply(
                lambda x: json.loads(x))
        mreport = mreport.set_index('Feature')
        all_feature = list(mreport.index.intersection(data.columns))
        if y is not None:
            y = pd.Series(y).copy()
            y.index = data.index
        all_report = pd.DataFrame()
        for feature in all_feature:
            feature_data = data[feature].copy()
            interval = mreport.ix[feature, 'Interval']

            if 'type' in mreport.columns:
                ftype = mreport.ix[feature, 'type']
            else:
                if isinstance(mreport.ix[feature, 'Interval'][0], list):
                    ftype = 'D'
                else:
                    ftype = 'C'

            result, bin_stat, bin_map = Bins.bin_replace(
                feature_data, interval, ftype, y)

            if y is not None:
                split_bin = []
                for m in range(0, len(bin_stat.index)):
                    interval_tmp = bin_stat['Interval'].iloc[m]
                    if isinstance(interval_tmp, list):
                        for n in interval_tmp:
                            if 'NaN' in n:
                                split_bin.append(bin_stat['Bin'].iloc[m])
                                break
                    else:
                        split_bin = split_bin + \
                            list(bin_stat.ix[bin_stat['Interval'] == 'NaN', 'Bin'])

                split_bin = list(set(split_bin))
                woe_result, woeiv, error_flag = Bins.woe_iv(
                    result, y,all_bin=list(bin_stat['Bin']),modified_flag = 1, split_bin=split_bin)
                bin_stat = bin_stat.drop(columns=[0, 1, 'total'])
                report = pd.merge(
                    bin_stat, woeiv, left_on='Bin', right_on='Bin')
            else:
                report = bin_stat.copy()

            report['Feature'] = feature
            all_report = all_report.append(report)

        if y is not None:
            all_report = all_report.sort_values(
                by=['iv', 'Feature', 'Bin'], ascending=[False, True, True])

        col_list = list(all_report.columns)
        col_list.remove('Feature')
        all_report = all_report[['Feature'] + col_list]

        return all_report

    @staticmethod
    def whole_bin_replace(data, report):
        '''
        Replace the raw data by bin
        :param data: DataFrame, the raw data
        :param report: DataFrame, the report used to replace
        :return: the replaced bin result
        '''

        data = data.copy()
        report = report.copy()

        all_feature = list(
            set(data.columns).intersection(set(report['Feature'])))
        all_result = {}

        for feature in all_feature:
            try:
                if feature in data.columns:
                    feature_report = report[report['Feature']
                                            == feature].copy()
                    feature_data = data[feature].copy()
                    feature_report.index = pd.Series(
                        range(0, len(feature_report.index))).apply(str)
                    feature_report['lower_json'] = feature_report['lower_json'].fillna(
                        'NaN')
                    feature_report['upper_json'] = feature_report['upper_json'].fillna(
                        'NaN')
                    feature_report['lower_json'] = feature_report['lower_json'].apply(
                        lambda x: json.loads(str(x)))
                    feature_report['upper_json'] = feature_report['upper_json'].apply(
                        lambda x: json.loads(str(x)))
                    if feature_report['type'].iloc[0] == 'C':
                        for t in range(0, len(feature_report.index)):
                            if pd.isnull(feature_report.ix[t, 'lower_json']):
                                feature_report.ix[t, 'Interval'] = 'NaN'
                            else:
                                feature_report.ix[t, 'Interval'] = pd.Interval(
                                    feature_report.ix[t, 'lower_json'], feature_report.ix[t, 'upper_json'])
                    else:
                        feature_report['Interval'] = feature_report['lower_json'].copy(
                        )

                    bin_map = Bins.__stat_to_map(
                        feature_report, feature_report['type'].iloc[0])
                    bin_interval = Bins.__stat_to_interval(
                        feature_report, feature_report['type'].iloc[0])
                    if feature_report['type'].iloc[0] == 'C':
                        interval_list = []
                        for i in range(0, len(bin_interval) - 1):
                            interval_list.append(pd.Interval(
                                bin_interval[i], bin_interval[i + 1]))
                        interval_feature_data = pd.cut(
                            feature_data,
                            pd.IntervalIndex(interval_list)).astype(
                            pd.Interval)
                        interval_feature_data = interval_feature_data.fillna(
                            'NaN')
                        result = interval_feature_data.apply(
                            lambda x: bin_map[x])
                    else:
                        feature_data = feature_data.apply(str)
                        feature_data = Bins.nan_normalize(feature_data)
                        feature_data = feature_data.fillna('NaN')

                        num_len = len(list(bin_map.values())[0]) - 1
                        result = feature_data.apply(
                            lambda x: bin_map.get(
                                x, 'B{:0>{}}'.format(
                                    0, num_len)))
                    all_result[feature] = result
            except Exception as e:
                print(e)
                raise Exception('Feature Error!')

        return pd.DataFrame(all_result)

    @staticmethod
    def whole_woe_replace(data, report):
        '''
        Replace the raw data by woe value
        :param data: DataFrame, the raw data
        :param report: DataFrame, the report used to replace
        :return: the replaced woe result
        '''

        data = data.copy()
        report = report.copy()
        all_feature = list(
            set(data.columns).intersection(set(report['Feature'])))
        bin_data = Bins.whole_bin_replace(data, report)

        all_woe_dict = {}
        for feature in all_feature:
            feature_report = report[report['Feature'] == feature]
            Bins.woe_iv(bin_data[feature], report=feature_report)
            woe_result, woeiv, error_flag = Bins.woe_iv(
                bin_data[feature], report=feature_report)
            all_woe_dict[feature] = woe_result

        return pd.DataFrame(all_woe_dict)[bin_data.columns]


class Evaluate(Widget):
    '''
    Evaluate Class which can be used to evaluate the Logistic Regression(score_card) result
    v1.0,copyright by AntiMage-Janhonho
    '''

    def __init__(self):
        pass

    @staticmethod
    def cal_auc_ks(y, score):
        '''
        Calculate the auc&ks value
        :param y: list_like,the real result
        :param score: list_like,the score
        :return: tuple of the result
        '''

        if Evaluate.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        fpr, tpr, thresholds = roc_curve(y, score)
        auc_value = auc(fpr, tpr)
        ks_value = max(tpr - fpr)

        return auc_value, ks_value, fpr, tpr, thresholds

    @staticmethod
    def cal_lift(y, score, bins=10):
        '''
        Calculate the lift result
        :param y: list_like,the real result
        :param score: list_like,the score
        :param bins: float or list_like, if float qcut, if list_like cut
        :return:
        '''

        if Evaluate.binary_check(y) is False:
            raise ValueError('Value Error of y!')

        df = pd.DataFrame({'y': y, 'score': score})
        try:
            bins = int(bins)
            grouping, retbin = pd.qcut(
                df['score'], bins, duplicates='drop', retbins=True)
            retbin = pd.Series(retbin).apply(lambda x: round(x, 4))
            retbin = retbin.drop_duplicates()
            grouping = pd.cut(df['score'], retbin)
        except Exception:
            grouping = pd.cut(df['score'], pd.IntervalIndex(bins))

        grouped = df['y'].groupby(grouping)
        df = grouped.apply(
            lambda x: {
                'total': x.count(),
                '1': x.sum()}).unstack()
        df = df.sort_index(ascending=False)

        df = df.reset_index()
        df['0'] = df['total'] - df['1']
        df['1/1_total'] = df['1'] / sum(df['1'])
        df['0/0_total'] = df['0'] / sum(df['0'])
        df['(1/1_total)_cumsum'] = df['1'].cumsum() / sum(df['1'])
        df['(0/0_total)_cumsum'] = df['0'].cumsum() / sum(df['0'])
        df['1/total'] = df['1'] / df['total']
        df['(1/total)_cumsum'] = df['1'].cumsum() / df['total'].cumsum()
        df['ks_score'] = (df['(1/1_total)_cumsum'] -
                          df['(0/0_total)_cumsum']).apply(np.abs)

        df = df[['score', '0', '1', 'total', '1/1_total', '0/0_total',
                 '(1/1_total)_cumsum', '(0/0_total)_cumsum', '1/total',
                 '(1/total)_cumsum', 'ks_score']]

        df = df.sort_values(by='score', ascending=True)

        return df

    @staticmethod
    def cal_confusion_matrix(y, y_predict):
        '''
        Caculate the confusion_matrix
        :param y: list_like,the real y
        :param y_predict: list_like,the predict y
        :return: DataFrame, the confusion matrix
        '''

        if Evaluate.binary_check(
                y) is False or Evaluate.binary_check(y_predict):
            raise ValueError('Value Error of y!')

        result = confusion_matrix(y, y_predict, labels=[0, 1])
        result = pd.DataFrame(result, index=['0', '1'], columns=['0', '1'])

        return result

    @staticmethod
    def plot_roc_curve(auc_value, fpr, tpr, save_path=os.getcwd()):
        '''
        :param auc_value: the calculated auc_value
        :param fpr: list_like,the calculated fpr
        :param tpr: list_like,the caculated tpr
        :return:
        '''

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.figure()
        plt.plot([0, 1], [0, 1], 'r--')
        plt.plot(fpr, tpr, label='ROC_curve')
        s = 'AUC:{:.4f}\nKS:{:.4f}'.format(auc_value, max(tpr - fpr))
        plt.text(0.6, 0.2, s, bbox=dict(facecolor='red', alpha=0.5))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC_curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(save_path, 'ROC_curve.png'))
        plt.close()

    @staticmethod
    def plot_lift_curve(input_df, save_path=os.getcwd()):
        '''
        :param input_df: DataFrame, the caculated ks_df
        :return:
        '''

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df = input_df.copy()
        df = df.sort_values(by='score', ascending=False)

        plt.figure()
        plt.plot(
            df['ks_score'].values,
            'r-*',
            label='KS_curve',
            lw=1.2)
        plt.plot(
            df['(0/0_total)_cumsum'].values,
            'g-*',
            label='(0/0_total)_cumsum',
            lw=1.2)
        plt.plot(
            df['(1/1_total)_cumsum'].values,
            'm-*',
            label='(1/1_total)_cumsum',
            lw=1.2)
        plt.plot([0, len(df.index) - 1], [0, 1], linestyle='--',
                 lw=0.8, color='k', label='Random_result')
        xtick = list(df['score'].apply(str))
        plt.xticks(np.arange(len(xtick)), xtick, rotation=60)
        plt.xlabel('Interval')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'KS_curve.png'))
        plt.close()

    @staticmethod
    def prob_to_score(prob):
        '''
        Calculate the score from the prob
        :param prob: list_like, the predict probability
        :return: list_like, the score
        '''
        prob_s = pd.Series(prob).copy()
        score = prob_s.apply(lambda x: -28.85 *
                             np.log(x / (1 - x + 1e-7)) + 481.86)

        return score

    @staticmethod
    def evaluate(y, score, save_path=os.getcwd(), bins=10, lift_flag=1):
        '''
        Combined function to calculate all metrics
        :param y: list_like,the real y
        :param score: list_like,the score
        :param save_path: str,the save root
        :param bin_num: the bin_num of the lift
        :return:
        '''
        auc_value, ks_value, fpr, tpr, thresholds = Evaluate.cal_auc_ks(
            y, score)

        Evaluate.plot_roc_curve(auc_value, fpr, tpr, save_path)
        lift_score = Evaluate.prob_to_score(score)
        lift_df = Evaluate.cal_lift(y, lift_score, bins)
        Evaluate.plot_lift_curve(lift_df, save_path)

        result_dict = {'auc_value': auc_value, 'ks_value': ks_value,
                       'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
                       'lift_df': lift_df}

        return result_dict


if __name__ == '__main__':

    root_dir = r'E:\jianhonghao\Desktop\Bin'
    data = pd.read_csv(os.path.join(root_dir, 'demo_data.csv'))
    x_df = data.drop(columns=['y'])
    x_df = x_df.applymap(str)
    y = data['y']

    # 1、快速的生成整个df的结果:

    # all_report是统计特征,
    # change_report是后续用于修改的一个文件，
    # woe_df和bin_df表示按照这样分箱woe替换结果和bin替换结果
    all_report, change_report, woe_df, bin_df, false_dict = Bins.generate_raw(
        x_df, y)

    all_report.to_csv(os.path.join(root_dir, 'report.csv'))
    change_report.to_csv(os.path.join(root_dir, 'creport.csv'))

    # 通过阅读all_report的结果，已经对creport.csv这个文件进行了修改,后面想看修改后整个结果，如果不满意继续修改creport.csv
    creport = pd.read_csv(os.path.join(root_dir, 'creport.csv'))
    report = Bins.mannual_rebin(x_df, creport, y)

    # 如果这个all_report是在之前某次生成的结果
    report = pd.read_csv(os.path.join(root_dir, 'report.csv'))

    # 对修改的分箱结果满意，那么分别看bin替代结果和woe替代结果
    all_result = Bins.whole_bin_replace(x_df, report)
    all_woe_result = Bins.whole_woe_replace(x_df, report)

    # 2、针对单个特征可以坐以下操作

    # 2.1 F1是连续变量

    # 2.1.1 不给出分箱的区间
    x = data['F1']

    # 初始化
    demo = Bins()
    # 启动运算
    demo.generate_bin_smi(x, y, ftype='C')
    # 输出分箱报告
    bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # 按照这种分箱将变量替换为箱子值
    bin_result = demo.value_to_bin(x)
    # 按照分箱替换的结果计算woe等
    woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # 按照woe结果将新数据进行woe替换
    woe_result, woe_report, error_flag = Bins.woe_iv(
        bin_result, report=woe_report)

    # 2.1.2 给出分箱的区间
    x = data['F1']
    demo = Bins()
    demo.generate_bin_smi(
        x, y, interval=[-4.222, -3.186, -2.992, -2.889, -2.883], ftype='C')
    bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # 按照这种分箱将变量替换为箱子值
    bin_result = demo.value_to_bin(x)
    # 按照分箱替换的结果计算woe等
    woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # 按照woe结果将新数据进行woe替换
    woe_result, woe_report, error_flag = Bins.woe_iv(
        bin_result, report=woe_report)

    # 2.2 F2是离散变量，但是本身有一定顺序，这里顺序可以指定，默认是根据名称

    # 2.2.1 不给出分箱的区间
    x = data['F2']
    demo = Bins()
    demo.generate_bin_smi(x, y, ftype='M')
    bin_stat, bin_interval, bin_map = demo.get_bin_info()
    result = demo.value_to_bin(x)
    # 按照这种分箱将变量替换为箱子值
    bin_result = demo.value_to_bin(x)
    # 按照分箱替换的结果计算woe等
    woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # 按照woe结果将新数据进行woe替换
    woe_result, woe_report, error_flag = Bins.woe_iv(
        bin_result, report=woe_report)

    # 2.2.2 给出分箱的区间
    x = data['F2']
    demo = Bins()
    demo.generate_bin_smi(x, y, interval=[['A01', 'A02']], ftype='M')
    bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # 按照这种分箱将变量替换为箱子值
    bin_result = demo.value_to_bin(x)
    # 按照分箱替换的结果计算woe等
    woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # 按照woe结果将新数据进行woe替换
    woe_result, woe_report, error_flag = Bins.woe_iv(
        bin_result, report=woe_report)

    # 2.3 F2是离散变量，但是本身不具有顺序

    # 2.3.1 不给出分箱的区间
    x = data['F2']
    demo = Bins()
    demo.generate_bin_smi(x, y, ftype='D')
    bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # 按照这种分箱将变量替换为箱子值
    bin_result = demo.value_to_bin(x)
    # 按照分箱替换的结果计算woe等
    woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # 按照woe结果将新数据进行woe替换
    woe_result, woe_report, error_flag = Bins.woe_iv(
        bin_result, report=woe_report)

    # 2.3.2 给出分箱的区间
    x = data['F2']
    demo = Bins()
    demo.generate_bin_smi(x, y, interval=[['A01', 'A02']], ftype='D')
    bin_stat, bin_interval, bin_map = demo.get_bin_info()
    # 按照这种分箱将变量替换为箱子值
    bin_result = demo.value_to_bin(x)
    # 按照分箱替换的结果计算woe等
    woe_result, woe_report, error_flag = Bins.woe_iv(bin_result, y)
    # 按照woe结果将新数据进行woe替换
    woe_result, woe_report, error_flag = Bins.woe_iv(
        bin_result, report=woe_report)

    # 3 进行简单的替换
    x = data['F1']
    bin_result, bin_stat, bin_map = Bins.bin_replace(
        x, interval=[-4.222, -3.186, -2.992, -2.889, -2.883], ftype='C')

    print('This is end')


