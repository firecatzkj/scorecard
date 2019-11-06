# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:30:09 2019

@author: zhaifeifei1
"""

#%%


import pandas as pd
import numpy as np

import sys
'''根据实际情况调整路径，防止SwapAuto模块无法读取'''
sys.path.append(r'E:\code\scorecard') 
from .SwapAuto_V2 import *


#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''         auc ks lift swap分析自动化输出                '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

f =open(r'E:\zhaifeifei1\Desktop\df_swap.csv')
df_swap =pd.read_csv(f)
f.close()




#####################################
''' 以下为参数输入 '''

'''
 如果需要切分数据集来看，则添加一个flag标签，否则不要添加flag字段  
'''

P1 ='magic_xgb_score' ##参数
P2 ='hujin_xgb_score' ##参数
y ='Y2'  ## Y值标签
P1_name ="pvalue_v402模型" ## 模型1的名称
P2_name ="pvalue_86模型"   ## 模型2的名称
GroupNums1 =10 #模型1的排序分组数
GroupNums2 =10 #模型2的排序分组数

CutNumP1 =1 ## swap统计分析表中模型1的切分组数
CutNumP2 =2 ## swap统计分析表中模型2的切分组数

filename ="test2.xls" ## 文档名称
outputh ="E:/code/scorecard/outputh/"

f =pf.Swap(df_swap,y,P1,P2,P1_name,P2_name,CutNumP1,CutNumP2,filename,outputh,GroupNums1,GroupNums2)


