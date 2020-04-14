# -*- coding: utf-8 -*-

'''
Process missing value
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from fancyimpute import KNN

# 用随机森林处理缺失值
def set_missing_rf(df):
    # 切割属性
    process_df = df.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]

    # 分成已知属性和未知属性
    known = process_df[process_df.MonthlyIncome.notnull()].values
    unknown = process_df[process_df.MonthlyIncome.isnull()].values

    # X为特征属性值
    X = known[:, 1:]
    # Y为结果标签
    Y = known[:, 0]

    # Random Forest training
    rfr = RandomForestRegressor(random_state=10, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, Y)

    # Random Forest testing
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print(predicted)

    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted

    # 删除缺失值（处理NumberOfDependents）
    df = df.dropna()

    # 删除重复项
    df = df.drop_duplicates()

    return df

# 用KNN处理缺失值
def set_missing_knn(df):

    df = df.dropna(subset=['NumberOfDependents'])

    # 分成已知属性和未知属性
    known = df[df.MonthlyIncome.notnull()]
    unknown = df[df.MonthlyIncome.isnull()]

    # take samples
    known = known.sample(10000)
    process_df = pd.concat([known, unknown], axis=0)

    # k-nn
    process_df = KNN(k=100).fit_transform(process_df)
    process_df = pd.DataFrame(process_df, columns=df.keys())
    df = df.dropna()
    df = pd.concat([df, process_df], axis=0)
    df['MonthlyIncome'] = df['MonthlyIncome'].round(0)

    # 删除重复项
    df = df.drop_duplicates()

    return df

if __name__ == '__main__':

    data = pd.read_csv('./data/cs-training.csv', index_col=0)
    # data.describe().to_csv('./data/DataDescribe.csv')

    # 补充缺失值（处理MonthlyIncome）
    # data = set_missing_rf(data)   # random forest

    data = set_missing_knn(data)    # K-NN

    data.to_csv('./data/MissingData.csv', index=False)