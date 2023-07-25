# Adult Census Income Prediction [ML_Internship_Project]

## WebApp: 
> Link: [Census Income Predictor](http://ec2-13-232-128-218.ap-south-1.compute.amazonaws.com:8080/)
---

## Domain:
Income of US Citizes, Census Bureau
  
## Objective:
This is a classification problem where we need to predict whether a person earns more than a sum of 50,000 USD anuually or not. This classification task is accomplished by using a CatBoost Classifier trained on the dataset extracted by Barry Becker from the 1994 Census database. The dataset contains about 33k records and 15 features which after all the implementation of all standard techniques like Data Cleaning, Feature Engineering, Feature Selection, Outlier Treatment, etc was feeded to our Classifier which after training and testing, was deployed in the form of a web application.

## Project Goal

This end-to-end project is made as a part of data science internship for [Ineuron.ai](https://ineuron.ai/).

• Working of web application

![Alt Text](https://drive.google.com/uc?export=download&id=1DHLG9_JiH3tDwJVvOVnPJjUJwfTYrvDA)

## Data and Source Description
#### Dataset : Adult Census Income from Kaggle
#### Source of the Data :  https://www.kaggle.com/datasets/overload10/adult-census-dataset

This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). The prediction task is to predict whether income exceeds $50k per year based on the provided. census data provided above. The Datasets consists of a list of records , each of which explains various features of a person along with his income per year. 

## Technical Aspects

The whole project has been divided into three parts. These are listed as follows :

• Data Preparation : This consists of storing our data into cassandra database and utilizing it, Data Cleaning, Feature Engineering, Feature Selection, EDA, etc.

• Model Development : In this step, we use the resultant data after the implementation of the previous step to cross validate our Machine Learning model and perform Hyperparameter optimization based on various performance metrics in order to make our model predict as accurate results as possible.

• Model Deployment : This step include creation of a front-end using Anvil, Flask and Heroku to put our trained model into production.


## Installation

The Code is written in Python 3.9. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:
```bash
pip install -r requirements.txt
```

### Run on your Local Machine

To run the flask server on your local machine, just run the following code in your command prompt in the project directory :
```bash
python app.py
```


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

• Python Programming Language: Used Python for developing the Adult Census Income Prediction model.

• AWS EC2: Deployed the web application on Amazon Web Services (AWS) Elastic Compute Cloud (EC2) for scalability.

• Flask: Employed Flask, a lightweight web framework in Python, for building the web application.

## Appendix

Link for youtube video regarding description of the project : https://youtu.be/ECzTXGu3Qt8

Linkedin Post : [Linkedin post URL](https://www.linkedin.com/posts/pritamchaudhari5_incomeprediction-datadrivendecisions-machinelearning-activity-7087327512178937856-TUQ2?utm_source=share&utm_medium=member_desktop)

## Author

- [Pritam Chaudhari](https://github.com/Pritamchaudhari5)

- [Linkedin](https://www.linkedin.com/in/pritamchaudhari5/)
    
## Evaluation

<b>Evaulation scores & Performance metrics obtained in our model:</b>

- F1 Score : 90.5823 

- Accuracy Score : 90.162 

- Mean Absolute Error : 9.838

## Conclusion
- Developed an end-to-end web application for Adult Census Income Prediction.
- Performed Exploratory Data Analysis (EDA) and preprocessed the data for model building.
- Built a machine learning model to predict whether a person's income is above or below 50K USD.
- Deployed the model on a user-friendly web interface for easy access and predictions.
- Achieved accurate predictions to assist concerned authorities in decision-making.

## Future Scope
- We have a large enough dataset, so we can use neural networks such as an artificial neural network to build a model which can result in better performance.