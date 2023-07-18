import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


dataset = pd.read_csv(r"D:\Intern projects\ML_Internship_Project__Adult_Census_Income_Prediction\Adult_Census_Income_Data\adult.csv")

# Dropping duplicates and handling missing values
dataset.drop_duplicates(inplace=True)
dataset['country'] = dataset['country'].replace(' ?',np.nan)
dataset['workclass'] = dataset['workclass'].replace(' ?',np.nan)
dataset['occupation'] = dataset['occupation'].replace(' ?',np.nan)
dataset.dropna(how='any',inplace=True)

for dataset in [dataset]:
    dataset.loc[dataset['country'] != ' United-States', 'country'] = 0
    dataset.loc[dataset['country'] == ' United-States', 'country'] = 1
    dataset.loc[dataset['race'] != ' White', 'race'] = 0
    dataset.loc[dataset['race'] == ' White', 'race'] = 1
    dataset.loc[dataset['workclass'] != ' Private', 'workclass'] = 0
    dataset.loc[dataset['workclass'] == ' Private', 'workclass'] = 1
    dataset.loc[dataset['hours-per-week'] <= 40, 'hours-per-week'] = 0
    dataset.loc[dataset['hours-per-week'] > 40, 'hours-per-week'] = 1


for col in dataset[dataset.columns]:  # To convert object data by label encoder
    if dataset[col].dtypes == 'object':
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
dataset = dataset.astype(int)
dataset=dataset.drop(["education"],axis=1)

X = dataset.drop(['salary'], axis=1)
y = dataset['salary']
y.value_counts(normalize=True)

# balancing the skewed data
from imblearn.over_sampling import RandomOverSampler 
rs = RandomOverSampler(random_state=30)

rs.fit(X,y)

X1,y1 = rs.fit_resample(X, y)

# Let's create a train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.30)


# Creaating inistance of RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

# Fitting the model
rfc.fit(X_train, y_train)

# Predicting over the test set and claculating F1
test_predict_rfc = rfc.predict(X_test)

# Pickle dump
pickle.dump(rfc, open('model1.pkl','wb'))

# Load the pickled model
pickle_model = pickle.load(open('model.pkl','rb'))

# Example predictions using the loaded model
result1 = pickle_model.predict([[28,1,212563,10,0,6,4,0,0,0,0,0,1]])[0]

result2 = pickle_model.predict([[50,0,83311,13,2,3,0,1,1,0,0,0,1]])[0]

print(result1)
print(result2)