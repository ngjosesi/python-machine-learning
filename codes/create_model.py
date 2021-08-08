import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import joblib

print('Creating the model...');

diabetesDF = pd.read_csv('../diabetes.csv')

print('Found '+ str(len(diabetesDF.index)) +' records in diabetes.csv...')

print('Transforming the data to numeric columns...')

dfLabel = np.asarray(diabetesDF['Outcome'])
dfData = np.asarray(diabetesDF.drop('Outcome',1))

means = np.mean(dfData, axis=0)
stds = np.std(dfData, axis=0)

print('Forming the Model...')

diabetesCheck = LogisticRegression(max_iter=1000)
diabetesCheck.fit(dfData, dfLabel)

#save model

joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')

print('Completed creation of model');