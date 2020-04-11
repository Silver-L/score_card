'''
LogisticRegression
'''

import pandas as pd
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm


if __name__ == '__main__':
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
