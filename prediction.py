import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import precision_score

lbl = preprocessing.LabelEncoder()

excel_data_df = pd.read_excel('Excel files/FormattedData(FrontBack).xlsx', sheet_name='Sheet1')
json_str = excel_data_df.to_json(orient="records")

data_df = excel_data_df.drop(columns=['url', 'Database type', 'Created at', 'Finalized at', 'Repo type'
                                      ])

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

# fit final model
model = SGDClassifier()
model.fit(X_train, y_train)


# xgr = xgb.XGBRegressor(random_state=2)
# xgr.fit(X_train, y_train)
# y_pred = xgr.predict(X_test)

# print('X test: ' + str(X_test))
# print('X train: ' + str(X_train))
# print('Y test: ' + str(y_test))
# print('Y train: ' + str(y_train))


y_pred = model.predict(X_train)

# calculate precision for binary classification problem
precision_macro = precision_score(y_test, y_pred, average='macro')
precision_micro = precision_score(y_test, y_pred, average='micro')
precision_weighted = precision_score(y_test, y_pred, average='weighted')

print('Accurancy score: ' + str(accuracy_score(y_train, y_pred)))
print('Precision Score (Macro): ' + str(precision_macro))
print('Precision Score (Micro): ' + str(precision_micro))
print('Precision Score (Weighted): ' + str(precision_weighted))


# Section for saving model
# filename = 'Model10k.sav'
# pickle.dump(model, open(filename, 'wb'))
