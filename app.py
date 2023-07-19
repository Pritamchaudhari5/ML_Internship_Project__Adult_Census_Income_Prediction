import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv(r"D:\Intern projects\ML_Internship_Project__Adult_Census_Income_Prediction\Adult_Census_Income_Data\adult.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

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

# balancing the skewed data
from imblearn.over_sampling import RandomOverSampler 
rs = RandomOverSampler(random_state=30)

rs.fit(X,y)

X,y = rs.fit_resample(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        marital_name = request.form['Marital Status']

        marital_value = 0
        
        if marital_name == 'Never-married':
            marital_value = 0
        elif marital_name == 'Married-civ-spouse':
            marital_value = 1
        elif marital_name == 'Divorced':
            marital_value = 2
        elif marital_name == 'Married-spouse-absent':
            marital_value = 3        
        elif marital_name == 'Separated':
            marital_value = 4
        elif marital_name == 'Married-AF-spouse':
            marital_value = 5
        elif marital_name == 'Widowed':
            marital_value = 6
        
    
        age_value = request.form['Age']

        workclass = request.form['Workclass']
        workclass_val = 1

        if workclass == 'Private':
            workclass_val = 1
        elif workclass != 'Private':
            workclass_val = 0

        fnlwgt = request.form['Fnlwgt']
        edu_num_value = request.form['Years of Education']

        occupation = request.form['Occupation Code']

        occupation_value = 0

        if occupation == 'Adm-clerical':
            occupation_value = 0
        elif occupation == 'Exec-managerial':
            occupation_value = 1
        elif occupation == 'Handlers-cleaners':
            occupation_value = 2
        elif occupation == 'Prof-specialty':
            occupation_value = 3
        elif occupation == 'Other-service':
            occupation_value = 4
        elif occupation == 'Sales':
            occupation_value = 5
        elif occupation == 'Craft-repair':
            occupation_value = 6
        elif occupation == 'Transport-moving':
            occupation_value = 7
        elif occupation == 'Farming-fishing':
            occupation_value = 8
        elif occupation == 'Machine-op-insect':
            occupation_value = 9
        elif occupation == 'Tech-support':
            occupation_value = 10
        elif occupation == 'Protective-serv':
            occupation_value = 11
        elif occupation == 'Armed-forces':
            occupation_value = 12
        elif occupation == 'Priv-house-serve':
            occupation_value = 13
        


        relationship = request.form['Relationship']   

        relationship_value = 0

        if relationship == 'Not-in-family':
            relationship_value = 0
        elif relationship == 'Husband':
            relationship_value = 1
        elif relationship == 'Wife':
            relationship_value = 2
        elif relationship == 'Own-child':
            relationship_value = 3
        elif relationship == 'Unmarried':
            relationship_value = 4
        elif relationship == 'Other-relative':
            relationship_value = 5

        race = request.form['Race']

        race_val = 1

        if race == 'white':
            race_val = 1
        elif race != 'white':
            race_val = 0

        sex = request.form['Sex']
        sex_val = 0

        if sex == 'Male':
            sex_val = 0
        elif sex == 'Female':
            sex_val = 1

        capital_gain = request.form["Capital Gain"]
        capital_loss = request.form['Capital Loss']

        hours = request.form['Hours']
        hours_value = 0
        if hours == "More-than-40":
            hours_value = 1
        elif hours == "Less-than-40":
            hours_value = 0

        country = request.form['Country']
        country_val = 1

        if country == ' United-States':
            country_val = 1
        elif country != ' United-States':
            country_val = 0

    
    features = [age_value, workclass_val, fnlwgt, edu_num_value, marital_value, 
                occupation_value, relationship_value, race_val, sex_val, capital_gain,
                 capital_loss, hours_value, country_val]
    
    #print(features)
    
    int_features = [int(a) for a in features]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #print(prediction)
    
    if prediction == 1:
        output = "Income is more than 50K"
    elif prediction == 0:
        output = "Income is less than 50K"
        
    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


