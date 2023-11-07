from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd 
import numpy as np
import os

df = pd.read_csv("3Ex1.csv")
X = df.iloc[:, :8]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1)

data_dir = "./data"
model_dir = './model'

os.makedirs(model_dir, exist_ok=True)


model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model,os.path.join(model_dir,"linear_model.joblib"))

print("Everthing done")