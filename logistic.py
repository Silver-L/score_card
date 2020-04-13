# -*- coding: utf-8 -*-

'''
LogisticRegression
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm


if __name__ == '__main__':

    # train
    train_data = pd.read_csv('./data/train_woe.csv')

    x_train = train_data.loc[:, train_data.columns != 'SeriousDlqin2yrs']
    y_train = train_data.loc[:, 'SeriousDlqin2yrs']

    x_train = x_train.drop(
        ['DebtRatio_woe', 'MonthlyIncome_woe', 'NumberOfOpenCreditLinesAndLoans_woe',
         'NumberRealEstateLoansOrLines_woe', 'NumberOfDependents_woe'], axis=1)

    X1 = sm.add_constant(x_train)
    logit = sm.Logit(y_train, X1)
    result = logit.fit()
    print(result.summary())

    # test
    test_data = pd.read_csv('./data/test_woe.csv')

    x_test = test_data.loc[:, test_data.columns != 'SeriousDlqin2yrs']
    y_test = test_data.loc[:, 'SeriousDlqin2yrs']

    x_test = x_test.drop(
        ['DebtRatio_woe', 'MonthlyIncome_woe', 'NumberOfOpenCreditLinesAndLoans_woe',
         'NumberRealEstateLoansOrLines_woe', 'NumberOfDependents_woe'], axis=1)

    X2 = sm.add_constant(x_test)
    predict =result.predict(X2)

    fpr, tpr, threshold = roc_curve(y_test, predict)
    rocauc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('真正率')
    plt.xlabel('假正率')
    plt.show()