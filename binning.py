# -*- coding: utf-8 -*-

'''
分箱
'''


import numpy as np
from scipy.stats import chi2
import pandas as pd
from pandas import DataFrame,Series
from binning.woebin import *
import scipy
import seaborn as sns


if __name__ == '__main__':
    data = pd.read_csv('./data/TrainData.csv')

    # 特征相关性计算
    corr = data.corr()  # 计算各变量的相关性系数
    xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']  # x轴标签
    yticks = list(corr.index)  # y轴标签
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1,
                annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})  # 绘制相关性系数热力图
    ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
    ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
    plt.show()

    # 分箱(卡方 or tree)
    cutoff = woebin(data, y='SeriousDlqin2yrs', method='tree')
    # print(cutoff["NumberOfTimes90DaysLate"]["woe"])

    feature_index = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
             'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
             'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']  # x轴的标签

    train_woe = woebin_ply(data, cutoff)
    woe_index = ['{}_{}'.format(s, 'woe') for s in feature_index]
    woe_index.insert(0, 'SeriousDlqin2yrs')
    train_woe = train_woe.loc[:, woe_index]
    train_woe.to_csv('./data/train_woe.csv', index=False)

    # test data
    test_data = pd.read_csv('./data/TestData.csv')
    test_woe = woebin_ply(test_data, cutoff)
    test_woe = test_woe.loc[:, woe_index]
    test_woe.to_csv('./data/test_woe.csv', index=False)


    # 求IV
    iv_list = []
    for i in range(10):
        iv_list.append(cutoff['{}'.format(feature_index[i])]['total_iv'][0])

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1, 1, 1)
    x = np.arange(len(feature_index))+1
    ax1.bar(x, iv_list, width=0.4)#生成柱状图
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_index, rotation=0, fontsize=7)
    ax1.set_ylabel('IV(Information Value)', fontsize=14)
    #在柱状图上添加数字标签
    for a, b in zip(x, iv_list):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
    plt.show()
