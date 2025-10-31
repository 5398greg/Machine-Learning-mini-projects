import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


sonar_data = pd.read_csv('./data/sonar_data.csv', header=None)

print(sonar_data.head())
print(sonar_data.shape)
print(sonar_data[60].value_counts())


# Under Sampling
mine = sonar_data[sonar_data[60] == 'M']
print(mine.shape)

mine_sample = mine.sample(n=97)
print(mine_sample.shape)

rock = sonar_data[sonar_data[60] == 'R']
print(rock.shape)

new_sonar_data = pd.concat([mine_sample, rock], axis=0)
print(new_sonar_data[60].value_counts())


# Label Encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(new_sonar_data[60])

new_sonar_data['Labels'] = labels

print(new_sonar_data.Labels.value_counts())

print(new_sonar_data)


# Splitting the data into features and labels X --> features Y --> labels
X = new_sonar_data.iloc[:, :60]
Y = new_sonar_data['Labels']

print(X)
print(Y)


# Splitting the Data into Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=21)

print(X.shape)
print(X_train.shape)
print(X_test.shape)


# Train the model (Support vector machine)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)


# Evaluating the accuracy score of the model
training_score = accuracy_score(model.predict(X_train), Y_train) * 100
print(f'Training score : {training_score: .2f}%')

testing_score = accuracy_score(model.predict(X_test), Y_test) * 100
print(f'Testing score : {testing_score: .2f}%')


# Creating a Predictive system based on input from user

data = (0.0134, 0.0172, 0.0178, 0.0363, 0.0444, 0.0744, 0.0800, 0.0456, 0.0368, 0.1250, 0.2405, 0.2325, 0.2523, 0.1472, 0.0669, 0.1100, 0.2353, 0.3282, 0.4416, 0.5167, 0.6508, 0.7793, 0.7978, 0.7786, 0.8587, 0.9321, 0.9454, 0.8645, 0.7220, 0.4850, 0.1357, 0.2951, 0.4715, 0.6036, 0.8083, 0.9870, 0.8800, 0.6411, 0.4276, 0.2702, 0.2642, 0.3342, 0.4335, 0.4542, 0.3960, 0.2525, 0.1084, 0.0372, 0.0286, 0.0099, 0.0046, 0.0094, 0.0048, 0.0047, 0.0016, 0.0008, 0.0042, 0.0024, 0.0027, 0.0041
        )

new_data = np.asarray(data).reshape(1, -1)

print("Object is ROCK") if model.predict(
    new_data) == 1 else print('Object is MINE')
