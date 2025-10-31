import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
