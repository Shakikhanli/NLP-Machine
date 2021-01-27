""" This file is for calculating weight of each feature of dataset  """
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


lbl = preprocessing.LabelEncoder()
excel_data_df = pd.read_excel('Excel files/FormattedData(FrontBack).xlsx', sheet_name='Sheet1')
json_str = excel_data_df.to_json(orient="records")


data_df = excel_data_df.drop(columns=['url', 'Created at', 'Finalized at', 'Repo type'])

data_df['ID of repository'] = lbl.fit_transform(data_df['ID of repository'].astype(str))
data_df['Name of repository'] = lbl.fit_transform(data_df['Name of repository'].astype(str))
data_df['Type'] = lbl.fit_transform(data_df['Type'].astype(str))
data_df['Framework'] = lbl.fit_transform(data_df['Framework'].astype(str))
data_df['Language'] = lbl.fit_transform(data_df['Language'].astype(str))
data_df['Database type'] = lbl.fit_transform(data_df['Database type'].astype(str))
data_df['ID of match'] = lbl.fit_transform(data_df['ID of match'].astype(str))
data_df['File structure'] = lbl.fit_transform(data_df['File structure'].astype(str))


X = data_df.drop('ID of match', axis=1)
y = data_df['ID of match']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# params = {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 4,
#           'n_estimators': 200, 'nthread': 4, 'silent': 1, 'subsample': 0.7}

parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 7,
              'min_child_weight': 4, 'n_estimators': 250, 'nthread': 4,
              'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7}



knn = KNeighborsClassifier()

# preds = {}
# for model_name, model in zip(['KNearestNeighbors'], [knn]):
#     model.fit(X_train, y_train)
#     preds[model_name] = model.predict(X_test)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
print("Score1: " + str(model.score(X_test, y_test)))

model = MultinomialNB()
model.fit(X_train, y_train)
print("Score Naive Bayes: " + str(model.score(X_test, y_test)))

model = AdaBoostClassifier()
model.fit(X_train, y_train)
print("Score Ada: " + str(model.score(X_test, y_test)))



xgr = xgb.XGBRegressor(random_state=2, **parameters)
xgr.fit(X_train, y_train)

for feature, importance in zip(list(X.columns), xgr.feature_importances_):
    print('Model weight for feature {}: {}'.format(feature, importance))

