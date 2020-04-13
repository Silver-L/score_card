# -*- coding: utf-8 -*-

'''
处理异常值
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv('./data/MissingData.csv')

    data = data[data['age'] > 0]

    # 剔除异常值
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    data = data[data['NumberOfTimes90DaysLate'] < 90]
    data = data[data['NumberOfTime60-89DaysPastDueNotWorse'] < 90]

    #变量SeriousDlqin2yrs取反
    data['SeriousDlqin2yrs'] = 1 - data['SeriousDlqin2yrs']

    # 调整低收入人群比例
    data_norm = data[data['MonthlyIncome'] > 10]
    data_low = data[data['MonthlyIncome'] < 10]
    data_low = data_low.sample(data_low.shape[0] // 10)
    data = pd.concat([data_norm, data_low], axis=0)

    # split data set
    X = data.iloc[:, 1:]
    Y = data['SeriousDlqin2yrs']

    #测试集占比30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
    print(clasTest)

    data.to_csv('./data/MissingData_erase_outlier.csv', index=False)
    train.to_csv('./data/TrainData.csv', index=False)
    test.to_csv('./data/TestData.csv', index=False)

    # # plot
    # outlier = data.iloc[:,[3, 7, 9]]
    # outlier.plot.box(grid = True, ylim = [0, 100])
    # plt.show()
