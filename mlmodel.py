import numpy as np
import pandas as pd  
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

consumption= pd.read_csv('FuelConsumption.csv')
print(consumption.head())
consumption.info();

features=['MODELYEAR','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']
X = consumption[features]
y = consumption['CO2EMISSIONS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

poly_features = PolynomialFeatures(degree=3, include_bias=False)
scaler = StandardScaler()
lin_reg = LinearRegression()
pipeline = make_pipeline(poly_features, scaler, lin_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

import pickle

filename = 'model.pickle'
pickle.dump(pipeline, open(filename, 'wb'))

