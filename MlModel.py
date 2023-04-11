import pandas as pd

import numpy as np

train = pd.read_csv(r"D:\College\sem 6\Mini Project\model\train.csv")

col=['ID','used_app_before','age_desc','relation']
train.drop(col, axis=1, inplace=True)
#train.to_csv("/Users/apple/Downloads/new mini/testedit.csv", index=False)

train['ethnicity'] = np.where((train['ethnicity'] == 'others') | (train['ethnicity'] == '?') | (train['ethnicity'] == 'Health care professional'), 'Others', train['ethnicity'])

# create a dictionary to map country names to integer values
ordinal_mapt = {country: index for index, country in enumerate(train['ethnicity'].unique())}
# replace country names with integer values using the ordinal_map dictionary
train['ethnicity'] = train['ethnicity'].map(ordinal_mapt)

train['Jaundice'] = (train['jaundice'] == 'yes')*1.0
train['Austim'] = (train['austim'] == 'yes')*1.0
train['Female'] = (train['gender'] == 'f')*1.0
train['Male'] = (train['gender'] == 'm')*1.0

col=['gender','jaundice','austim']
train.drop(col, axis=1, inplace=True)
#train.to_csv("/Users/apple/Downloads/new mini/testedit.csv", index=False)

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

# create a dictionary to map country names to integer values
ordinal_map = {country: index for index, country in enumerate(train['contry_of_res'].unique())}
# replace country names with integer values using the ordinal_map dictionary
train['contry_of_res'] = train['contry_of_res'].map(ordinal_map)

col=['result']
train.drop(col, axis=1, inplace=True)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Split the data into features and target
y = train['Class/ASD']
X = train.drop('Class/ASD', axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import AdaBoostClassifier
base_estimator = DecisionTreeClassifier(max_depth=1)

# create the AdaBoost classifier
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10)

# fit the classifier to the training data
clf.fit(X_train, y_train)

# predict the class labels for new data
y_pred = clf.predict(X_test)
accc = accuracy_score(y_test,y_pred)
print("Accuracy:", accc)

import pickle
pickle.dump(clf, open('model.pkl', 'wb'))
