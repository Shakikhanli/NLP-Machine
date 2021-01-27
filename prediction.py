"""Making prediction in different algorithms"""


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import label_ranking_average_precision_score

from sklearn.inspection import permutation_importance

lbl = preprocessing.LabelEncoder()

excel_data_df = pd.read_excel('Excel files/FinalDataset.xlsx', sheet_name='Sheet1')
json_str = excel_data_df.to_json(orient="records")

data_df = excel_data_df.drop(columns=['url', 'Created at', 'Finalized at', 'Repo type'
                                      ])

data_df['ID of repository'] = lbl.fit_transform(data_df['ID of repository'].astype(str))
data_df['Name of repository'] = lbl.fit_transform(data_df['Name of repository'].astype(str))
data_df['Type'] = lbl.fit_transform(data_df['Type'].astype(str))
data_df['Framework'] = lbl.fit_transform(data_df['Framework'].astype(str))
data_df['Language'] = lbl.fit_transform(data_df['Language'].astype(str))
data_df['Database type'] = lbl.fit_transform(data_df['Database type'].astype(str))
data_df['ID of match'] = lbl.fit_transform(data_df['ID of match'].astype(str))
data_df['IDs of developers'] = lbl.fit_transform(data_df['IDs of developers'].astype(str))
data_df['File structure'] = lbl.fit_transform(data_df['File structure'].astype(str))

# Separate the features and the response
X = data_df.drop('ID of match', axis=1)
y = data_df['ID of match']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)




# fit final model
"""***************************************************************************************************"""

# model = GradientBoostingClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_train)
#
# print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
# print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
# print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
# print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
# print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
# print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
# print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)))
#
# for feature, importance in zip(list(X.columns), model.feature_importances_):
#     print('Model weight for feature {}: {}'.format(feature, importance))
# print(' ')
#
# print('\n')

# """***************************************************************************************************"""
#
# try:
#     model = SGDClassifier()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_train)
#
#
#     print('Result of SGD classifier')
#
#     print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
#     print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
#     print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
#     print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
#     print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
#     print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
#     print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)))
#     print(label_ranking_average_precision_score(y_train, y_pred))
#     print('\n')
# except:
#     print('SGD classifier error' + '\n')
# """*******************************************************************************"""
#
# try:
#     model = RidgeClassifier()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_train)
#
#     importance = model.coef_
#
#     print('Result of Ridge classifier')
#
#     print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
#     print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
#     print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
#     print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
#     print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
#     print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
#     print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
#     for i, v in enumerate(importance):
#         print('Feature: %0d, Score: %.5f' % (i, v))
#     print('\n')
# except:
#     print('Ridge classifier error' + '\n')
# """*******************************************************************************"""

try:


    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X)

    print('Result of Multinomial NB')

    imps = permutation_importance(model, X_train, y_train)
    print(imps.importances_mean)

    print('Accurancy score: ' + str(accuracy_score(y, y_pred)))
    print('Precision score: ' + str(precision_score(y, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y, y_pred, average='macro', zero_division=1)) + '\n')

    print('\n')
except:
    print('Multinomial NB error' + '\n')
# """*******************************************************************************"""
#
# model = LogisticRegression(max_iter=100)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_train)
# importance = model.coef_[0]
#
# print('Result of Logistic Regression')
#
# print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
# print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
# print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
# print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
# print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
# print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
# print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
# for i, v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (i, v))
# print('\n')
# """*******************************************************************************"""
#
# model = KNeighborsClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_train)
#
# print('Result of K nearest neighbor')
#
# print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
# print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
# print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
# print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
# print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
# print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
# print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
#
# print(label_ranking_average_precision_score(y_train, y_pred))
#
# """*******************************************************************************"""

try:
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Ada boost')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Ada boost classifier not worked' + '\n')
"""*******************************************************************************"""

try:
    model = AdaBoostRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Ada boost regeressor')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Ada boost regressor not worked' + '\n')
"""*******************************************************************************"""

try:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Random forest classifier')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Random forest error' + '\n')
"""*******************************************************************************"""

try:
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Random forest regressor')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Random forest regressor error' + '\n')
"""*******************************************************************************"""

try:
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Decision tree')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Decision tree ERROR' + '\n')
"""*******************************************************************************"""

try:
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Decision tree regressor')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Decision tree regressor ERROR' + '\n')
"""*******************************************************************************"""

try:
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Extra tree classfier')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Extra tree classifier ERROR' + '\n')
"""*******************************************************************************"""

try:
    model = ExtraTreesRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    print('Result of Extra tree regressor')

    print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
    print('Precision score: ' + str(precision_score(y_train, y_pred, average='micro', zero_division=1)))
    print('Recall score: ' + str(recall_score(y_train, y_pred, average='micro', zero_division=1)))
    print('F1 score: ' + str(f1_score(y_train, y_pred, average='micro', zero_division=1)) + '\n')
    print('Precision score (macro): ' + str(precision_score(y_train, y_pred, average='macro', zero_division=1)))
    print('Recall score (macro): ' + str(recall_score(y_train, y_pred, average='macro', zero_division=1)))
    print('F1 score (macro): ' + str(f1_score(y_train, y_pred, average='macro', zero_division=1)) + '\n')
    for feature, importance in zip(list(X.columns), model.feature_importances_):
        print('Model weight for feature {}: {}'.format(feature, importance))
    print(' ')
except:
    print('Extra tree regressor ERROR' + '\n')

# xgr = xgb.XGBRegressor(random_state=2)
# xgr.fit(X_train, y_train)
# y_pred = xgr.predict(X_test)


# y_pred = model.predict(X_test)
#
#
# print('Accurancy score: ' + str(accuracy_score(y_test, y_test)))
# print('Precision Score (Macro): ' + str(precision_score(y_train, y_pred, average='macro')))
# print('Precision Score (Micro): ' + str(precision_score(y_train, y_pred, average='micro')))
# print('Precision Score (Weighted): ' + str(precision_score(y_train, y_pred, average='weighted')))


# Section for saving model
# filename = 'Model10k.sav'
# pickle.dump(model, open(filename, 'wb'))
