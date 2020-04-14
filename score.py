# -*- coding: utf-8 -*-
'''
评分卡
'''

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scorecardpy as sc
import numpy as np

if __name__ == '__main__':
    train_data = pd.read_csv('./data/TrainData.csv')

    # 分箱(卡方 or tree)
    break_list = {'DebtRatio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.7],
                  'NumberRealEstateLoansOrLines': [0, 1, 2, 3]}

    cutoff = sc.woebin(train_data, y='SeriousDlqin2yrs', method='chimerge', breaks_list=break_list)
    # print(cutoff["NumberOfTimes90DaysLate"]["woe"])

    feature_index = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
             'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
             'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']  # x轴的标签

    train_woe = sc.woebin_ply(train_data, cutoff)
    woe_index = ['{}_{}'.format(s, 'woe') for s in feature_index]
    woe_index.insert(0, 'SeriousDlqin2yrs')
    train_woe = train_woe.loc[:, woe_index]

    # test data
    test_data = pd.read_csv('./data/TestData.csv')
    test_woe = sc.woebin_ply(test_data, cutoff)
    test_woe = test_woe.loc[:, woe_index]

    # train lr
    x_train = train_woe.loc[:, train_woe.columns != 'SeriousDlqin2yrs']
    y_train = train_woe.loc[:, 'SeriousDlqin2yrs']

    x_train = x_train.drop(
        ['DebtRatio_woe', 'MonthlyIncome_woe', 'NumberOfOpenCreditLinesAndLoans_woe',
         'NumberRealEstateLoansOrLines_woe', 'NumberOfDependents_woe'], axis=1)

    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
    lr.fit(x_train, y_train)
    print("lr coefficient: {}".format(lr.coef_))
    print("lr intercept: {}".format(lr.intercept_))

    # test
    x_test = test_woe.loc[:, test_woe.columns != 'SeriousDlqin2yrs']
    y_test = test_woe.loc[:, 'SeriousDlqin2yrs']

    x_test = x_test.drop(
        ['DebtRatio_woe', 'MonthlyIncome_woe', 'NumberOfOpenCreditLinesAndLoans_woe',
         'NumberRealEstateLoansOrLines_woe', 'NumberOfDependents_woe'], axis=1)

    train_pred = lr.predict_proba(x_train)[:, 1]
    test_pred = lr.predict_proba(x_test)[:, 1]

    # # lr (statsmodels version)
    # import statsmodels.api as sm
    # X1 = sm.add_constant(x_train)
    # logit = sm.Logit(y_train, X1)
    # result = logit.fit()
    # print(result.summary())
    # X2 = sm.add_constant(x_test)
    # predict = result.predict(X2)

    # score ------
    card = sc.scorecard(cutoff, lr, x_train.columns, points0=600, odds0=1 / 20, pdo=20, basepoints_eq0=False)
    column = ['basepoints', 'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
              'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']

    if not (os.path.exists('./data/card')): os.makedirs(os.path.join('./data/card'))
    for i in card.keys():
        card[i].to_csv('./data/card/{}.csv'.format(i), index=False)

    # performance ks & roc ------
    train_perf = sc.perf_eva(y_train, train_pred, title="train")
    test_perf = sc.perf_eva(y_test, test_pred, title="test")

    # credit score
    train_score = sc.scorecard_ply(train_data, card, print_step=0, only_total_score=False)
    test_score = sc.scorecard_ply(test_data, card, print_step=0, only_total_score=False)

    # save score
    if not (os.path.exists('./data/score')): os.makedirs(os.path.join('./data/score'))
    train_score.to_csv('./data/score/train_score.csv', index=False)
    test_score.to_csv('./data/score/test_score.csv', index=False)

    # psi
    sc.perf_psi(
        score={'train': pd.DataFrame(train_score.loc[:, 'score']), 'test': pd.DataFrame(test_score.loc[:, 'score'])},
        label={'train': y_train, 'test': y_test}
    )
