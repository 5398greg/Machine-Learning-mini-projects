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


# # Label Encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(new_sonar_data[60])

new_sonar_data['Labels'] = labels

print(new_sonar_data.Labels.value_counts())

print(new_sonar_data)


# # Splitting the data into features and labels X --> features Y --> labels
X = new_sonar_data.iloc[:, :60]
Y = new_sonar_data['Labels']

print(X)
print(Y)


# # Splitting the Data into Train and Test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=21)

print(X.shape)
print(X_train.shape)
print(X_test.shape)


# # Train the model (Support vector machine)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)


# # Evaluating the accuracy score of the model
training_score = accuracy_score(model.predict(X_train), Y_train) * 100
print(f'Training score : {training_score: .2f}%')

testing_score = accuracy_score(model.predict(X_test), Y_test) * 100
print(f'Testing score : {testing_score: .2f}%')


# # Creating a Predictive system based on input from user

data = (0.0286, 0.0453, 0.0277, 0.0174, 0.0384, 0.0990, 0.1201, 0.1833, 0.2105, 0.3039, 0.2988, 0.4250, 0.6343, 0.8198, 1.0000, 0.9988, 0.9508, 0.9025, 0.7234, 0.5122, 0.2074, 0.3985, 0.5890, 0.2872, 0.2043, 0.5782, 0.5389, 0.3750, 0.3411, 0.5067, 0.5580, 0.4778, 0.3299, 0.2198, 0.1407, 0.2856, 0.3807, 0.4158, 0.4054, 0.3296, 0.2707, 0.2650, 0.0723, 0.1238, 0.1192, 0.1089, 0.0623, 0.0494, 0.0264, 0.0081, 0.0104, 0.0045, 0.0014, 0.0038, 0.0013, 0.0089, 0.0057, 0.0027, 0.0051, 0.0062
        )

new_data = np.asarray(data).reshape(1, -1)

print("Object is ROCK") if model.predict(
    new_data) == 1 else print('Object is MINE')
