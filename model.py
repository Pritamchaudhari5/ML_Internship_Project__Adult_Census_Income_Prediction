import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

dataset = pd.read_csv(r"D:\Intern projects\ML_Internship_Project__Adult_Census_Income_Prediction\Adult_Census_Income_Data\adult.csv")

# Dropping duplicates and handling missing values
dataset.drop_duplicates(inplace=True)
dataset['country'] = dataset['country'].replace(' ?', np.nan)
dataset['workclass'] = dataset['workclass'].replace(' ?', np.nan)
dataset['occupation'] = dataset['occupation'].replace(' ?', np.nan)
dataset.dropna(how='any', inplace=True)

# Encoding categorical features
for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])

# Creating a new DataFrame with appropriate column names
dataset_encoded = dataset.astype(int)
dataset_encoded = dataset_encoded.drop(["education"], axis=1)

X = dataset_encoded.drop(['salary'], axis=1)
y = dataset_encoded['salary']

# Let's create a train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

# Creating an instance of RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

# Fitting the model
rfc.fit(X_train, y_train)

# Predicting over the test set and calculating F1
test_predict_rfc = rfc.predict(X_test)

# Pickle dump
pickle.dump(rfc, open('model1.pkl', 'wb'))

# Load the pickled model
pickle_model = pickle.load(open('model2.pkl','rb'))

# Example predictions using the loaded model
result1 = pickle_model.predict([[28, 1, 212563, 10, 0, 6, 4, 0, 0, 0, 0, 0, 1]])[0]
result2 = pickle_model.predict([[50, 0, 83311, 13, 2, 3, 0, 1, 1, 0, 0, 0, 1]])[0]

print(result1)  # Output: 0 or 1 (predicted class for the first example)
print(result2)  # Output: 0 or 1 (predicted class for the second example)
