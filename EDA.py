'''
EDA(Exploratory Data Analysis) 探索性分析
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    data = pd.read_csv('./data/MissingData_erase_outlier.csv')
    # data = data.dropna()

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.hist(x = data['age'], bins=30, edgecolor='k', linewidth=1)
    plt.xlabel('age')

    plt.subplot(2, 2, 2)
    data['age'].plot(kind='kde')
    plt.xlim([0,100])
    plt.xlabel('age')

    plt.subplot(2, 2, 3)
    plt.hist(x = data['MonthlyIncome'], bins=6000)
    plt.xlim([-10000, 60000])
    plt.xlabel('MonthlyIncome')

    plt.subplot(2, 2, 4)
    data['MonthlyIncome'].plot(kind='kde')
    plt.xlim([-20000, 80000])
    plt.xlabel('MonthlyIncome')

    plt.grid()
    plt.show()