import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import joblib

print('Predicting using saved model...');

print('Reading the CSV to predict per row...');

diabetesDF = pd.read_csv('../diabetes.csv')

dfLabel = np.asarray(diabetesDF['Outcome'])
dfData = np.asarray(diabetesDF.drop('Outcome',1))

print('Loading the Model...');

diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')

print('Performing prediction...');

accuracyModel = diabetesLoadedModel.score(dfData, dfLabel)

print("accuracy = ", accuracyModel * 100,"%")

