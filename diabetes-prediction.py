# importng dependecies
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
