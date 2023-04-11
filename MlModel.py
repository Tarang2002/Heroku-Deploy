import pandas as pd
import tensorflow as tf
import numpy as np

train = pd.read_csv(r"D:\College\sem 6\Mini Project\model\train.csv")
test= pd.read_csv(r"D:\College\sem 6\Mini Project\model\test.csv")

col=['ID','used_app_before','age_desc','relation']
train.drop(col, axis=1, inplace=True)
#train.to_csv("/Users/apple/Downloads/new mini/testedit.csv", index=False)

train['ethnicity'] = np.where((train['ethnicity'] == 'others') | (train['ethnicity'] == '?') | (train['ethnicity'] == 'Health care professional'), 'Others', train['ethnicity'])

# create a dictionary to map country names to integer values
ordinal_map = {country: index for index, country in enumerate(train['ethnicity'].unique())}
# replace country names with integer values using the ordinal_map dictionary
train['ethnicity'] = train['ethnicity'].map(ordinal_map)

train.head()

train['Jaundice'] = (train['jaundice'] == 'yes')*1.0
train['Austim'] = (train['austim'] == 'yes')*1.0
train['Female'] = (train['gender'] == 'f')*1.0
train['Male'] = (train['gender'] == 'm')*1.0

col=['gender','jaundice','austim']
train.drop(col, axis=1, inplace=True)
#train.to_csv("/Users/apple/Downloads/new mini/testedit.csv", index=False)

train.head()

from sklearn.preprocessing import MinMaxScaler

# sample data
X = train[['age']]
# Initialize the Scaler
scaler = MinMaxScaler()

# Fit the Scaler to the data
scaler.fit(X)

# Transform the data
X_scaled = scaler.transform(X)

# Output the scaled data
print(X_scaled)

train['age']=X_scaled

X = train[['result']]
# Initialize the Scaler
scaler = MinMaxScaler()

# Fit the Scaler to the data
scaler.fit(X)

# Transform the data
X_scaled = scaler.transform(X)

# Output the scaled data
print(X_scaled)

train['result']=X_scaled

train.head()

# create a dictionary to map country names to integer values
ordinal_map = {country: index for index, country in enumerate(train['contry_of_res'].unique())}
# replace country names with integer values using the ordinal_map dictionary
train['contry_of_res'] = train['contry_of_res'].map(ordinal_map)

col=['result','contry_of_res']
train.drop(col, axis=1, inplace=True)

train.head()

import pandas as pd

# read the dataset

# create the correlation matrix
corr_matrix = train.corr()

# print the correlation matrix
print(corr_matrix)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Split the data into features and target
y = train['Class/ASD']
X = train.drop('Class/ASD', axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import xgboost as xgb
from xgboost import plot_tree

# convert data to DMatrix format required by XGBoost
dmatrix = xgb.DMatrix(data=X_train, label=y_train)

# build the XGBoost classifier
params = {'objective': 'binary:logistic', 'eval_metric': 'error','gamma':0.2}
model = xgb.train(dtrain=dmatrix, params=params, num_boost_round=10)

# predict the class labels for new data

new_dmatrix = xgb.DMatrix(data=X_test)
predictions = model.predict(new_dmatrix)
predictions = [round(pred) for pred in predictions]
acccc = accuracy_score(y_test,predictions)
print("Accuracy:", acccc)

import pickle
pickle.dump(model, open('model.pkl', 'wb'))