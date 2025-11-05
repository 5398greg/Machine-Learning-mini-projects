# importng dependecies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# getting the diabetes data
diabetes_data = pd.read_csv('./data/diabetes.csv')

# Viewing the data and some few details
print(diabetes_data)
print(diabetes_data.shape)
print(diabetes_data['Outcome'].value_counts())

# Undersampling the data
# reducing the non_diabetic to 268

non_diabetic = diabetes_data[diabetes_data['Outcome'] == 0]
diabetic = diabetes_data[diabetes_data['Outcome'] == 1]

print(non_diabetic.shape)
print(diabetic.shape)

new_none_diabetic = non_diabetic.sample(n=268)
print(new_none_diabetic.shape)

new_diabetes_data = pd.concat([new_none_diabetic, diabetic], axis=0)
print(new_none_diabetic)
print(new_none_diabetic.shape)

# Splitting the data features and Outcome where X --> Features and Y --> Outcome
X = new_diabetes_data.drop('Outcome', axis=1)
Y = new_diabetes_data['Outcome']

print(X)
print(Y)

# Standardising the Features data
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(X.std())
print(Y)

# Splitting the data to Train and test for our model training
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=9)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Using a logistic reggression model to Train our data
model = LogisticRegression()
model.fit(X_train, Y_train)

# Checking the accuracy score on training and testing data
training_score = accuracy_score(model.predict(X_train), Y_train) * 100
print(f'Your Training score is: {training_score: .2f}%')

testing_score = accuracy_score(model.predict(X_test), Y_test) * 100
print(f'Your Testing score is: {testing_score: .2f}%')


# Building a system for users input

input_data = (13, 145, 82, 19, 110, 22.2, 0.245, 57)

acceptable_input_data = scaler.transform(np.asarray(input_data).reshape(1, -1))

print('You ARE diabetic') if model.predict(acceptable_input_data)[0] == 1 else print(
    'You are NOT diabetic')

print(model.predict(acceptable_input_data))
