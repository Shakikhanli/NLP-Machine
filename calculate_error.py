"""This file is for calculating mean square error"""
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

count = 0
lbl = preprocessing.LabelEncoder()

excel_data_df = pd.read_excel('Excel files/FormattedData(FrontBack).xlsx', sheet_name='Sheet1')
json_str = excel_data_df.to_json(orient="records")

data_df = excel_data_df.drop(columns=['url', 'Database type', 'Created at', 'Finalized at', 'Repo type'])

data_df['ID of repository'] = lbl.fit_transform(data_df['ID of repository'].astype(str))
data_df['Name of repository'] = lbl.fit_transform(data_df['Name of repository'].astype(str))
data_df['Type'] = lbl.fit_transform(data_df['Type'].astype(str))
data_df['Framework'] = lbl.fit_transform(data_df['Framework'].astype(str))
data_df['Language'] = lbl.fit_transform(data_df['Language'].astype(str))
data_df['ID of match'] = lbl.fit_transform(data_df['ID of match'].astype(str))
data_df['File structure'] = lbl.fit_transform(data_df['File structure'].astype(str))

# Separate the features and the response
X = data_df.drop('ID of match', axis=1)
y = data_df['ID of match']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


""" First regression"""

# Instantiate an XGBRegressor
xgr = xgb.XGBRegressor(random_state=2)

# Fit the classifier to the training set
xgr.fit(X_train, y_train)
y_pred = xgr.predict(X_test)

print("%.10f" % mean_squared_error(y_test, y_pred))


"""*****************************************************"""

# params = {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 4,
#           'n_estimators': 250, 'nthread': 4, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7}

# """Finding better parameters"""
# # Various hyper-parameters to tune
# xgb1 = xgb.XGBRegressor()
# parameters = {'colsample_bytree': [0.7],
#                          'learning_rate': [0.03, 0.05, 0.07],
#                          'max_depth': [5, 6, 7], 'min_child_weight': [4],
#                          'n_estimators': [250], 'nthread': [4],
#                          'objective': ['reg:linear'], 'silent': [1],
#                          'subsample': [0.7]}
#
# xgb_grid = GridSearchCV(xgb1, parameters, cv = 2, n_jobs = 5, verbose=True)
#
# print(xgb_grid.fit(X_train, y_train))
"""***********************************************************"""

"""Try regression with new parameters"""
# # Try again with new params
# xgr1 = xgb.XGBRegressor(random_state=2, **params)
#
# # Fit the classifier to the training set
# xgr1.fit(X_train, y_train)
#
# y_pred = xgr1.predict(X_test)
#
# print("%.10f" % mean_squared_error(y_test, y_pred))

