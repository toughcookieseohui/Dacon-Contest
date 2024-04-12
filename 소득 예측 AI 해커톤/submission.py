''' 불러오기 '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")



''' EDA '''
# print(test.shape, train.shape) --> (10000, 22) (20000, 23)
# display(sample_submission)

train = train.drop("ID", axis=1)
train_income = train.pop("Income")
test_id = test.pop("ID")
# print(train.shape, train_income.shape, test.shape, test_id.shape) --> (20000, 21) (20000,) (10000, 21) (10000,)
# print(train.columns)
# --> 'Age', 'Gender', 'Education_Status', 'Employment_Status',
#        'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race',
#        'Hispanic_Origin', 'Martial_Status', 'Household_Status',
#        'Household_Summary', 'Citizenship', 'Birth_Country',
#        'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status',
#        'Gains', 'Losses', 'Dividends', 'Income_Status'
train_num = train[train.describe().columns].copy()
train_obj = train[train.columns.difference(train.describe().columns)].copy()
test_num = test[test.describe().columns].copy()
test_obj = test[test.columns.difference(test.describe().columns)].copy()

''' 전처리 '''
# 인코딩
label_encoder = LabelEncoder()
for col in train_obj.columns:
  train_obj[col] = label_encoder.fit_transform(train_obj[col])
for col in test_obj.columns:
  test_obj[col] = label_encoder.fit_transform(test_obj[col])

# 데이터프레임 통합
train = pd.concat([train_num, train_obj], axis=1)
test = pd.concat([test_num, test_obj], axis=1)

# 새로운 특성 생성
# train['New_Feature_1'] = train['Age'] + train['Gains'] + train['Losses']
train['New_Feature_2'] = train['Education_Status'] + train['Working_Week (Yearly)'] + train['Gains']
train['New_Feature_3'] = train['Industry_Status'] + train['Occupation_Status'] + train['Dividends']
train['New_Feature_4'] = train['Race'] + train['Losses'] + train['Gains']
train['New_Feature_5'] = train['Tax_Status'] + train['Dividends'] + train['Age']
train['New_Feature_6'] = train['Working_Week (Yearly)'] * train['Employment_Status']
train['New_Feature_7'] = train['Gains'] - train['Losses']
train['New_Feature_8'] = train['Age'] * train['Education_Status']
train['New_Feature_9'] = train['Race'] + train['Tax_Status']

# test['New_Feature_1'] = test['Age'] + test['Gains'] + test['Losses']
test['New_Feature_2'] = test['Education_Status'] + test['Working_Week (Yearly)'] + test['Gains']
test['New_Feature_3'] = test['Industry_Status'] + test['Occupation_Status'] + test['Dividends']
test['New_Feature_4'] = test['Race'] + test['Losses'] + test['Gains']
test['New_Feature_5'] = test['Tax_Status'] + test['Dividends'] + test['Age']
test['New_Feature_6'] = test['Working_Week (Yearly)'] * test['Employment_Status']
test['New_Feature_7'] = test['Gains'] - test['Losses']
test['New_Feature_8'] = test['Age'] * test['Education_Status']
test['New_Feature_9'] = test['Race'] + test['Tax_Status']


''' ML 검증 '''
X_train, X_val, y_train, y_val = train_test_split(train, train_income, test_size=0.3, random_state=500)

# model = RandomForestRegressor()
model = GradientBoostingRegressor()
# model = XGBRegressor()
# model = LinearRegression()
# model = make_pipeline(PolynomialFeatures(2), LinearRegression())
# model = SVR(kernel='rbf')
# model = MLPRegressor(hidden_layer_sizes=(90,), activation='relu', solver='adam', max_iter=1000)


model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
val_rmse = sqrt(mean_squared_error(y_val, val_predictions))
print("검증 세트의 RMSE:", val_rmse)

''' ML 예측 '''
test_predictions = model.predict(test)
test_rmse = sqrt(mean_squared_error(sample_submission['Income'], test_predictions))
print("테스트 데이터의 RMSE:", test_rmse)

''' 제출 '''
# RandomForestRegressor
# 검증 세트의 RMSE: 614.2429881082817
# 테스트 데이터의 RMSE: 718.9688788660668

# GradientBoostingRegressor
# 검증 세트의 RMSE: 597.0377799691837
# 테스트 데이터의 RMSE: 658.2582114830661

# XGBRegressor
# 검증 세트의 RMSE: 615.0131999697509
# 테스트 데이터의 RMSE: 710.7950372142986

# LinearRegression
# 검증 세트의 RMSE: 641.3211737416274
# 테스트 데이터의 RMSE: 637.0182374013845 -> 0.21, 3000

''' 제출 '''
result = pd.DataFrame({'ID': test_id, 'Income': test_predictions})
result.to_csv('submission.csv', index=False)
result
