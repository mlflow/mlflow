#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

#Loading Data
main_df = pd.read_csv('heart.csv')
df = main_df.copy()

#Data Preprocessing
cols = ['trtbps', 'chol', 'thalachh', 'oldpeak', 'age']
for col in cols:
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col] - minimum)/ (maximum - minimum)

X = df.drop("output", axis=1)
y = df['output']

#Spliting Training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=42)

#Training model
model_lg = LogisticRegression(max_iter=120,random_state=0, n_jobs=20)
model_lg.fit(X_train, y_train)

#Saving the model
filename = 'heart-attack-prediction.pkl'
pickle.dump(model_lg, open(filename, 'wb'))
print ('Model File Created...')