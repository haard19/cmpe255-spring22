import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv(r"C:\Users\Checkout\Documents\CMPE 255\cmpe255-spring22\assignment1\Levels_Fyi_Salary_Data.csv")
df.head()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df['year'] = df['timestamp'].dt.year
df.drop(["timestamp"], axis = 1, inplace=True)

df.drop_duplicates(inplace=True)

df = df.drop(columns = ['Race', 'Education', 'cityid', 'otherdetails', 'Some_College', 'Masters_Degree', 'Bachelors_Degree', 
                            'Doctorate_Degree', 'Race_Asian', 'Race_White', 'Race_Two_Or_More', 'level', 'tag', 
                            'Race_Black', 'Race_Hispanic', 'Highschool', 'rowNumber', 'dmaid', 'gender'])

# Since 'company' is an important feature for us let's replace the null values with "NA".

df['company'] = df['company'].fillna("NA")

## LabelEncoder and scaling

labelencoder_company = LabelEncoder()
df['company'] = labelencoder_company.fit_transform(df['company'])

labelencoder_location = LabelEncoder()
df['location'] = labelencoder_location.fit_transform(df['location'])

labelencoder_title = LabelEncoder()
df['title'] = labelencoder_title.fit_transform(df['title'])

## Modeling

X, y = df.iloc[:, :-1], df.iloc[:, -1]

p = 0.7

X_train = X.sample(frac = p, random_state = 42)
X_test = X.drop(X_train.index)
y_train = y.sample(frac = p, random_state = 42)
y_test = y.drop(y_train.index)

model = Sequential()
model.add(Dense(15, input_dim = len(X_train.iloc[0]), activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(loss='sparse_categorical_crossentropy', optimizer = "adam", metrics = 'mean_squared_error')

history = model.fit(X_train, y_train, epochs = 100)

y_pred = model.predict(X_test)
y_pred_flat = y_pred.flatten()

## Q1: How much you would get if I join for a position based on number of experiences and location?

position = 'Data Scientist'
experience = 3
location = 'Seattle, WA'

def predict_salary_q1(position, experience, location):
  yearsatcompany, bonus = 2, 10000
  basesalary, stockgrantvalue = 155000, 20000
  year = 2022
  company = 'Amazon'
  
  tempComp = labelencoder_company.transform([company])[0]
  pos = labelencoder_title.transform([position])[0]
  loc = labelencoder_location.transform([location])[0]

  data = {'year': [year],'company': [tempComp],'position': [pos],'location': [loc],'experience': [experience], "yearsatcompany": [yearsatcompany], 
            'basesalary': [basesalary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]}
  df = pd.DataFrame(data)
  y_test_pred_sal = model.predict(df).flatten()
  print('Predicted Salary: ', y_test_pred_sal[0])

predict_salary_q1(position, experience, location)

## Q2: How much you would get if you accept an offer for a position from X company based on number of experiences and location?

position = 'Data Scientist'
company = 'Oracle'
experience = 3
location = 'Seattle, WA'

def predict_salary_q2(company, position, experience, location):
  yearsatcompany, bonus = 2, 10000
  basesalary, stockgrantvalue = 155000, 20000  
  year = 2022
  
  tempComp = labelencoder_company.transform([company])[0]
  pos = labelencoder_title.transform([position])[0]
  loc = labelencoder_location.transform([location])[0]

  data = {'year':[year],'company':[tempComp],'position':[pos],'location':[loc],'experience':[experience], "yearsatcompany": [yearsatcompany], 
            'basesalary': [basesalary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]}
  df = pd.DataFrame(data)
  y_test_pred_sal = model.predict(df).flatten()
  print('Predicted Salary: ', y_test_pred_sal[0])

predict_salary_q2(company, position, experience, location)

## Q3: How much you will be getting for a position after Y years joining to X company?

position = 'Data Scientist'
company = 'Oracle'
yearsatcompany = 5

def predict_salary_q3(yearsatcompany, position, company):
  experience, location = 3, 'Seattle, WA'
  bonus, year = 10000, 2022
  basesalary, stockgrantvalue = 155000, 20000

  tempComp = labelencoder_company.transform([company])[0]
  pos = labelencoder_title.transform([position])[0]
  loc = labelencoder_location.transform([location])[0]

  
  data = {'year':[year],'company':[tempComp],'position':[pos],'location':[loc],'experience':[experience], "yearsatcompany": [yearsatcompany], 
            'basesalary': [basesalary], 'stockgrantvalue': [stockgrantvalue], 'bonus': [bonus]}
  df = pd.DataFrame(data)
  y_test_pred_sal = model.predict(df).flatten()
  print('Predicted Salary: ', y_test_pred_sal[0])

predict_salary_q3(yearsatcompany, position, company)