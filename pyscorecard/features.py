# -*- coding: utf8 -*-
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import GradientBoostingClassifier
from functools import reduce
from copy import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from .Tool import Bins
from .metrics.variable import psi_var


class FeatureSelect:
    """
        # 特征选择
        random_forest,lasso,gradient_boosting_rf特征选择方法
    """

    def __init__(self, df: pd.DataFrame, y: str, dtypes: dict=None, do_binning=False, df_test=None):
        """
        :param df: 进行单变量评估的数据集
        :param y: 因变量
        :param dtypes: 数据类型dict var_name: var_type
        """
        self.do_binning = do_binning
        self.df_test = df_test
        self.df = df
        self.y = y
        # print(df.columns, "KKKKKK")
        dtypes_all = self.get_all_dtypes(df, y, dtypes)
        if do_binning:
            self.df, self.x, self.y, self.var_dummied, self.iv_df = self.prepare_data_binning(df, y, dtypes_all)
        else:
            self.df, self.x, self.y, self.var_dummied, self.iv_df = self.prepare_data_no_binning(df, y, dtypes_all)

    def get_all_dtypes(self, df: pd.DataFrame, y: str, dtypes: dict):
        """
        对数据集的数据类型进行自动推导
        :param df:
        :param y:
        :param dtypes
        :return:
        """
        dtypes_all = df.drop(y, axis=1).dtypes
        dtypes_all = {str(x): str(y) for x, y in dtypes_all.to_dict().items()}
        dtypes_all_final = {}
        for k, v in dtypes_all.items():
            if str(v).startswith("int"):
                dtypes_all_final[k] = "int"
            elif str(v).startswith("float"):
                dtypes_all_final[k] = "float"
            else:
                dtypes_all_final[k] = "str"
        if dtypes:
            dtypes_all_final.update(dtypes)
        return dtypes_all_final

    def prepare_data_binning(self, df: pd.DataFrame, y: str, dtypes: dict):
        """
        对数据做自动分箱处理
        :param df:
        :param y:
        :param dtypes:
        :return:
        """
        x = [q for q in dtypes.keys()]
        y_data = df[y]
        var_dummied = []
        bb = Bins()
        all_report, change_report, woe_df, bin_df, false_dict = bb.generate_raw(df[x], y_data, MDP=True)
        # 计算iv
        iv_df = all_report[["Feature", "iv"]].drop_duplicates()
        iv_df.rename({"Feature": "var_code"}, axis=1, inplace=True)
        iv_df = iv_df.sort_values(by="iv", ascending=False)
        iv_df["iv_rank"] = [x + 1 for x in list(range(len(iv_df)))]

        # test_woe
        if self.df_test is not None:
            test_woed = Bins.whole_woe_replace(self.df_test, all_report)

            # print(woe_df.columns)
            # print(test_woed.columns)
            # 计算psi
            psi_df = []
            for var in x:
                tmp = {}
                try:
                    tmp["var_code"] = var
                    tmp["psi"] = psi_var(woe_df[var],test_woed[var])
                except:
                    tmp["var_code"] = var
                    tmp["psi"] = None
                psi_df.append(copy(tmp))
            psi_df = pd.DataFrame(psi_df)
            # 合并
            psi_iv_df = pd.merge(iv_df, psi_df, on="var_code", how="inner")
        else:
            psi_iv_df = iv_df

        ############
        del bb
        woe_df_copy = woe_df.copy()
        woe_df_copy[y] = y_data

        # 后续优化 inf
        pd.options.mode.use_inf_as_na = True
        woe_df_copy = woe_df_copy.fillna(0)
        woe_df_copy = woe_df_copy.round(6)
        # print(np.isinf(woe_df_copy).any())
        x_successed = list(woe_df_copy.columns)
        x_successed.remove(y)
        pd.options.mode.use_inf_as_na = False
        return woe_df_copy, x_successed, y, var_dummied, psi_iv_df

    def prepare_data_no_binning(self, df: pd.DataFrame, y: str, dtypes: dict):
        """
        :param df: 进行单变量评估的数据集
        :param y: 因变量
        :param dtypes: 数据类型dict var_name: var_type
        """
        df = df.copy()
        x = list(dtypes.keys())

        """
        类型转换:
            int -> int
            float -> float
            str: 
                类别>5 kmeans => dummy
                类别<=5 dummy
        """
        var_dummied = []
        for sub in dtypes.items():
            var_name = sub[0]
            var_type = sub[1]
            if var_type == "int":
                try:
                    df[var_name] = df[var_name].astype("int")
                except ValueError as e:
                    print(var_name, e)
                    x.remove(var_name)
                continue
            elif var_type == "float":
                try:
                    df[var_name] = df[var_name].astype("float")
                except ValueError as e:
                    print(var_name, e)
                    x.remove(var_name)
                continue
            elif var_type == "str":
                if len(df[var_name].unique()) > 5:
                    df[var_name] = self.category_var_cluster(df, var_name, y)
                    #print(df[var_name].unique(), "XXXXXXXXXXXX")
                var_dummied.append(var_name)
                var_dummy = pd.get_dummies(df[var_name], prefix=var_name)
                del df[var_name]
                x.remove(var_name)
                for q in var_dummy.columns:
                    x.append(q)
                df = pd.concat([df, var_dummy], axis=1)
                continue

        """
        缺失值填充
        """
        df = df.fillna(df.median())
        return df, x, y, var_dummied, None

    def category_var_cluster(self, df: pd.DataFrame, x: str, y: str):
        """
        对类别过多的变量,进行聚类,按照聚类之后的结果,进行dummy
        https://blog.csdn.net/xyisv/article/details/82430107
        确定K => 数据转换 => 返回dummy数据集 ------> 后续会和主数据集合并
        :param df: 数据集
        :param x: 需要进行聚类的类别型变量
        :return: 按照聚类结果tramsform的数据集
        """
        sub_df = df[[x, y]].copy()
        le = LabelEncoder()
        sub_df[x] = le.fit_transform(sub_df[x].fillna("Missing").values)

        best_k = self._find_that_k(df, x, y)
        # print(x, "best_k:", best_k)
        estimator = KMeans(n_clusters=best_k)
        result = estimator.fit_transform(sub_df[[x, y]])
        # print(result.shape, "XXAAQQ")
        # print(pd.DataFrame(result).head(5))
        result = pd.DataFrame(result)
        var_transformed = result.apply(lambda x: x.idxmin(), axis=1)
        return var_transformed

    def _find_that_k(self, df: pd.DataFrame, x: str, y: str, min_pct_diff=0.1):
        """
        寻找kmeans最合适的K
        :param df:
        :param x:
        :param y:
        :param min_pct_diff
        :return:
        """
        sub_df = df[[x, y]].copy()
        le = LabelEncoder()
        sub_df[x] = le.fit_transform(sub_df[x].fillna("Missing").values)

        # '利用SSE选择k'
        SSE = []  # 存放每次结果的误差平方和
        for k in range(1, 9):
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            ss = estimator.fit_transform(sub_df[[x, y]])
            SSE.append(estimator.inertia_)

        pct = []
        for i in range(len(SSE)):
            if i == 0:
                pct.append(1)
            else:
                pct.append(round(SSE[i] / SSE[i - 1], 6))
        #print(SSE)
        #print(pct)
        best_k = None
        for j in range(1, len(pct), 1):
            diff = abs(pct[j] - pct[j - 1])
            if diff < min_pct_diff:
                best_k = j
                break
        # 如果上面的循环都跑完了,还么找到最合适的K,建议删除这个变量
        if best_k:
            return best_k
        else:
            print("Cant't find best K for kmeans check your variable")
            return None

    def random_forest(self, grid_search=False, param=None):
        """
        : param grid_search ：bool, 是否进行网格搜索进行调参，默认为False
        : param param ：dict, 进行网格搜索时的参数设定，默认为None
        : return imp_df ：dataframe, 包含特征、特征重要性、排序 三列
        """
        df, selected, target = self.df, self.x, self.y
        optimal_params = {'n_estimators': 30, 'max_depth': 2, 'min_samples_leaf': int(len(df) * 0.03)}
        if grid_search:
            print('entered random_forest cv')
            if param == None:
                param = {'n_estimators': [10, 20, 30, 40], 'max_depth': [2, 3, 4, 5], \
                         'min_samples_leaf': [int(len(df) * 0.05), int(len(df) * 0.06), int(len(df) * 0.07)]}
            clf_obj = RandomForestClassifier()
            cv_obj = GridSearchCV(estimator=clf_obj, param_grid=param, scoring='roc_auc', cv=5)
            print(df[selected])
            cv_obj.fit(df[selected], df[target])
            optimal_params = cv_obj.best_params_
            # print(optimal_params)
            # optimal_params = cv_obj.cv_results_['params'][cv_obj.cv_results_['mean_test_score'].argmax()]
        rf = RandomForestClassifier(criterion='entropy', \
                                    n_estimators=optimal_params['n_estimators'], \
                                    max_depth=optimal_params['max_depth'], \
                                    min_samples_leaf=optimal_params['min_samples_leaf'])

        rf.fit(df[selected], df[target].astype(int))
        importance = rf.feature_importances_
        importance = importance.tolist()
        imp_df = pd.DataFrame([selected, importance]).T
        imp_df.columns = ['var_code', 'random_forest_score']
        imp_df.loc[:, 'random_forest_rank'] = imp_df.random_forest_score.rank(ascending=False)
        return imp_df

    def lasso(self, alpha=1):
        """
        : param df : dataframe, woe后的dataframe
        : param selected : list, 列名list
        : param target : string, 标签列名
        : param alpha ：float, lasso回归的alpha值，默认为None
        : return imp_df ：dataframe, 包含特征、回归系数、排序 三列
        """
        df, selected, target = self.df, self.x, self.y
        cv_lasso_obj = LassoCV(cv=10, normalize=True)
        cv_lasso_obj.fit(df[selected], df[target])
        alpha = cv_lasso_obj.alpha_
        lasso_obj = Lasso(alpha=alpha)
        lasso_obj.fit(df[selected], df[target])
        lasso_obj_coef = lasso_obj.coef_.tolist()
        imp_df = pd.DataFrame([selected, lasso_obj_coef]).T
        imp_df.columns = ['var_code', 'lasso_coef']
        imp_df.loc[:, 'lasso_rank'] = abs(imp_df.lasso_coef).rank(ascending=False)
        return imp_df

    def gradient_boosting_rf(self, grid_search=False, param=None):
        """
        : param df : dataframe, woe后的dataframe
        : param selected : list, 列名list
        : param target : string, 标签列名
        : param grid_search ：bool, 是否进行网格搜索进行调参，默认为False
        : param param ：dict, 进行网格搜索时的参数设定，默认为None
        : return imp_df ：dataframe, 包含特征、特征重要性、排序 三列
        """
        df, selected, target = self.df, self.x, self.y
        optimal_params = {'learning_rate': 0.01, 'n_estimators': 100, 'max_depth': 4,
                          'min_samples_leaf': int(len(df) * 0.03)}
        if grid_search:
            print('gradient_boosting_rfCV')
            if param == None:
                param = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 110], 'max_depth': [3, 4, 5],
                         'min_samples_leaf': [int(len(df) * 0.03), int(len(df) * 0.04), int(len(df) * 0.05)]}
            clf_obj = GradientBoostingClassifier()
            cv_obj = GridSearchCV(clf_obj, param, scoring='roc_auc', cv=5)
            cv_obj.fit(df[selected].astype(float), df[target].astype(int))
            optimal_params = cv_obj.best_params_
        gbdt_obj = GradientBoostingClassifier(learning_rate=optimal_params['learning_rate'], \
                                              max_depth=optimal_params['max_depth'],
                                              min_samples_leaf=optimal_params['min_samples_leaf'], \
                                              n_estimators=optimal_params['n_estimators'])
        # gbdt = GradientBoostingClassifier(init=None,learning_rate=0.1,loss='deviance',max_depth=3, max_features=None,
        # max_leaf_nodes=None,min_samples_leaf=1,min_samples_split=2,
        # min_weight_fraction_leaf=0.0,n_estimators=100,random_state=None,
        # subsample=1.0,verbose=0,warm_start=False)
        gbdt_obj.fit(df[selected].astype(float), df[target].astype(float))
        importance = gbdt_obj.feature_importances_
        importance = importance.tolist()
        imp_df = pd.DataFrame([selected, importance]).T
        imp_df.columns = ['var_code', 'gbdt_score']
        imp_df.loc[:, 'gbdt_rank'] = imp_df.gbdt_score.rank(ascending=False)
        return imp_df

    def xgboost(self, grid_search=False, param=None):
        '''
        : param df : dataframe, woe后的dataframe
        : param selected : list, 列名list
        : param target : string, 标签列名
        : param grid_search ：bool, 是否进行网格搜索进行调参，默认为False
        : param param ：dict, 进行网格搜索时的参数设定，默认为None
        : return imp_df ：dataframe, 包含特征、特征重要性、排序 三列
        '''
        df, selected, target = self.df, self.x, self.y
        optimal_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 4,
                          'min_samples_leaf': int(len(df) * 0.03)}

        if grid_search:
            print('xgboost_CV')
            xgtrain = xgb.DMatrix(df[selected].values, label=df[target].values)
            cvresult = xgb.cv(optimal_params, xgtrain, num_boost_round=optimal_params['n_estimators'], nfold=5,
                             metrics='auc', early_stopping_rounds=100)
            print(cvresult.shape[0])
            optimal_params['n_estimators'] = cvresult.shape[0]
            if param == None:
                param = {'learning_rate': [0.01, 0.1], 'n_estimators': optimal_params['n_estimators'],
                         'max_depth': [3, 4, 5],
                         'min_samples_leaf': [int(len(df) * 0.03), int(len(df) * 0.04), int(len(df) * 0.05)]}
            xgb1 = XGBClassifier()
            cv_obj = GridSearchCV(xgb1, param, scoring='roc_auc', cv=5)
            cv_obj.fit(df[selected].astype(float), df[target].astype(int))
            optimal_params = cv_obj.best_params_
        xgb_obj = XGBClassifier(learning_rate=optimal_params['learning_rate'], \
                                max_depth=optimal_params['max_depth'],
                                min_samples_leaf=optimal_params['min_samples_leaf'], \
                                n_estimators=optimal_params['n_estimators'])

        xgb_obj.fit(df[selected].astype(float), df[target].astype(float))
        importance = xgb_obj.feature_importances_
        importance = importance.tolist()
        imp_df = pd.DataFrame([selected, importance]).T
        imp_df.columns = ['var_code', 'xgb_score']
        imp_df.loc[:, 'xgb_rank'] = imp_df.xgb_score.rank(ascending=False)
        return imp_df

    def psi_data(self):
        """
        输出3列
            varcode: 变量名
            psi: psi
            psi_rank: psi排名
        :return:
        """
        # todo: psi计算

    def iv_data(self, all_report: pd.DataFrame):
        """
        输出3列
            varcode: 变量名
            iv: iv
            iv_rank: iv排名
        :return:
        """
        sub = all_report[["Feature", "iv"]].drop_duplicates()
        sub.rename({"Feature": "var_code"}, axis=1, inplace=True)
        sub = sub.sort_values(by="iv", ascending=False)
        sub["iv_rank"] = [x + 1 for x in list(range(len(sub)))]
        return sub

    def recover_dummy_rank(self, res: pd.DataFrame):
        """
        对单一算法的变量重要性结果中dummy的变量:
            - 同一个变量的dummy取排序最高的作为该变量的重要性排序
            - 用原本变量名来替代排序最高的dummy变量
            - 删除其余的dummy变量
        :param res: 单个算法出来的result 比如lasso
        :return: df
        """
        # TODO: 待实现
        raise NotImplementedError

    def get_combine_result(self, grid_search=False, param=None):
        """
        依次调用随机森林,lasso, gdbt,xgboost获取算法筛选变量的结果
        :param df:
        :param selected:
        :param target:
        :param grid_search:
        :param param:
        :return:
        """
        # 随机森林
        rf_res = self.random_forest(grid_search, param)

        # lasso
        lasso_res = self.lasso()

        # gdbt
        gdbt_res = self.gradient_boosting_rf(grid_search, param)

        # xgboost
        xgb_res = self.xgboost(grid_search, param)

        dfs = [rf_res, lasso_res, gdbt_res, xgb_res]

        # IV
        if self.do_binning:
            dfs.append(self.iv_df)

        res = reduce(lambda left, right: pd.merge(left, right, on='var_code'), dfs)
        return res

    def filter_by_treshhold(self, combine_result: pd.DataFrame, condition: dict, how: str) -> list:
        """
        按照条件过滤变量
        combine_result = self.get_combine_result(xx,xx,xx)
        :param combine_result: dataframe
        :param condition:
            condition中op对应的操作有:
            (需要严格按照下面的格式来! *_*!!)
                eq: =
                ne: !=
                le: <=
                lt: <
                ge: >=
                gt: >
            condition =  {
                "var1": {"op": "ge", "v": xxxx},
                "var2": {"op": "le", "v": xxxx},
                "var3": {"op": "eq", "v": xxxx},
                "var4": {"op": "gt", "v": xxxx},
                "var5": {"op": "lt", "v": xxxx},
                "var6": {"op": "ge", "v": xxxx},
            }
        :param: how: 这个参数和pd.merge的how相同, 交集: inner, 并集:outer
        :return:
        """
        dfs = []
        for var in condition.keys():
            op = condition[var]["op"]
            v = condition[var]["v"]
            filter_list = getattr(combine_result[var], op)(v)
            df = combine_result[filter_list]
            dfs.append(df)
        final = reduce(lambda left, right: pd.merge(left, right, on='var_code', how=how), dfs)
        return final


if __name__ == '__main__':
    df = pd.read_csv("/Users/clay/Code/scorecard/learn/qhxlj.csv")

    # df = df[[
    #     "id_ethnic", "id_sex", "is_op_type", "y"
    # ]]

    # var_types = pd.read_excel(
    #     "/Users/clay/Code/scorecard/learn/代码及输出/输出/variables_summary.xlsx",
    #     sheet_name="Sheet1"
    # )[[
    #     "var_code", "数据类型"
    # ]]
    # dtypes = {}
    # for i, j in zip(var_types["var_code"], var_types["数据类型"]):
    #     dtypes[i] = j

    dtypes = {
        "id_ethnic": "str",
        "id_sex": "str",
        "is_op_type": "str",
        "ava_storage_size": "float",
        "contact1a_relationship": "float",
        "family_live_type": "str",
        "household_type": "str",
        "other_fee_1": "float",
        "recharge_hour_std": "float",
        "trade_type": "str",
        "work_times": "float",
    }
    fs = FeatureSelect(df, "y", None, do_binning=True)
    res = fs.get_combine_result()
    res.to_csv("../out/feature_new22222.csv", index=False, encoding="gbk")


    # import xgboost
    # xgboost.plot_importance()
