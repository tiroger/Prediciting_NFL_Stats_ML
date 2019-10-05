# Next play Predictions

# Importing libraries and modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Opening dataset
play_by_play = pd.read_csv('resources/final_table_for_model_build.csv')
play_by_play.head()


# Building models without weather data

# Keeping only play and run plays
play_by_play.play_type.unique()
only_productive_plays = play_by_play[(play_by_play.play_type == 'pass') | (play_by_play.play_type == 'run')]
# Removing duplicates
only_productive_plays.drop_duplicates()
only_productive_plays.head()


# Data Pre-processing

only_productive_plays.columns
# Selecting features
X = only_productive_plays[['yardline_100', 'qtr', 'half_seconds_remaining',
       'game_seconds_remaining', 'down', 'ydstogo',
       'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
       'score_differential']]
y = only_productive_plays.play_type

X.head()

y.head()
print(X.shape)
print(y.shape)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Saving training testing datasets as csv for future use
X_train.to_csv('resources/X_train', index=False)
X_test.to_csv('resources/X_test', index=False)
y_train.to_csv('resources/y_train', index=False)
y_test.to_csv('resources/y_test', index=False)

# Scaling the dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
X_scaler = MinMaxScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

########################################
###         LOGISTIC REGRESSION      ###
########################################

from sklearn.linear_model import LogisticRegression
LogReg_classifier = LogisticRegression(multi_class='multinomial', solver ='newton-cg')
LogReg_classifier
LogReg_classifier.fit(X_train, y_train)
LogReg_train_score = LogReg_classifier.score(X_train, y_train)
LogReg_test_score = LogReg_classifier.score(X_test, y_test)
print(f"Training Data Score: {LogReg_train_score}")
print(f"Training Data Score: {LogReg_test_score}")

predictions = LogReg_classifier.predict(X_test)
predictions_LogReg = pd.DataFrame({"Prediction": predictions, "Actual": y_test})
predictions_LogReg.head()

 # Coefficients
LogReg_coefficients = LogReg_classifier.coef_
LogReg_coefficients_array = coefficients[0]
LogReg_coefficients_array
# Calculating odds ratio
LogReg_odds_ratio = np.exp(LogReg_coefficients_array)
LogReg_odds_ratio

labels = list(X.columns)
labels
values = [LogReg_coefficients_array[i] for i in range(len(LogReg_odds_ratio))]

LogReg_odds_ratio_df = pd.DataFrame(values, index=labels, columns=['LogReg_coefficients']).reset_index()
LogReg_odds_ratio_df.rename(columns={'index': 'feature'}, inplace=True)

LogReg_odds_ratio_df.sort_values('feature', ascending=False)

# Import the necessaries libraries
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()
import plotly.express as px
fig = px.bar(LogReg_odds_ratio_df, x='feature', y="LogReg_coefficients", orientation='v')
fig.show()


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(LogReg_classifier, random_state=1).fit(X_train,y_train)
eli5.show_weights(perm, feature_names = X.columns.tolist())

########################################
###      Bagging meta-estimator      ###
########################################

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
Bag_model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
Bag_model.fit(X_train, y_train)
Bag_model.score(X_test, y_test)

Bag_Model_train_score = Bag_model.score(X_train, y_train)
Bag_Model_test_score = Bag_model.score(X_test, y_test)
print(f"Training Data Score: {Bag_Model_train_score}")
print(f"Testing Data Score: {Bag_Model_test_score}")

########################################
###             AdaBoost             ###
########################################

from sklearn.ensemble import AdaBoostClassifier
AdaB_model = AdaBoostClassifier(random_state=1)
AdaB_model.fit(X_train, y_train)
AdaB_model.score(X_test, y_test)

AdaB_model_train_score = AdaB_model.score(X_train, y_train)
AdaB_model_test_score = AdaB_model.score(X_test, y_test)
print(f"Training Data Score: {AdaB_model_train_score}")
print(f"Testing Data Score: {AdaB_model_test_score}")

########################################
###             XGBoost              ###
########################################

import xgboost as xgb
XGB_model = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
XGB_model.fit(X_train, y_train)
XGB_model.score(X_test ,y_test)

XGB_model_train_score = XGB_model.score(X_train, y_train)
XGB_model_test_score = XGB_model.score(X_test, y_test)
print(f"Training Data Score: {XGB_model_train_score}")
print(f"Testing Data Score: {XGB_model_test_score}")

########################################
###         Decision Tree            ###
########################################

from sklearn import tree
DecTree_model = tree.DecisionTreeClassifier()
DecTree_model.fit(X_train, y_train)
DecTree_model.score(X_test, y_test)

DecTree_model_train_score = DecTree_model.score(X_train, y_train)
DecTree_model_test_score = DecTree_model.score(X_test, y_test)
print(f"Training Data Score: {DecTree_model_train_score}")
print(f"Testing Data Score: {DecTree_model_test_score}")

########################################
###         RANDOM FOREST            ###
########################################

from sklearn.ensemble import RandomForestClassifier
RandFor_model = RandomForestClassifier(n_estimators=100)
RandFor_model.fit(X_train, y_train)
RandFor_model.score(X_test, y_test)

RandFor_model_train_score = RandFor_model.score(X_train, y_train)
RandFor_model_test_score = RandFor_model.score(X_test, y_test)
print(f"Training Data Score: {RandFor_model_train_score}")
print(f"Testing Data Score: {RandFor_model_test_score}")


########################################
###             SUMMARY              ###
########################################

algorithms = ['Logistic Regression', 'Bagging meta-estimator', 'AdaBoost', 'XGBoost', 'Decision Tree', 'Random Forest']
training_scores = [LogReg_train_score, Bag_Model_train_score, AdaB_model_train_score, XGB_model_train_score, DecTree_model_train_score, RandFor_model_train_score]
testing_scores = [LogReg_test_score, Bag_Model_test_score, AdaB_model_test_score, XGB_model_test_score, DecTree_model_test_score, RandFor_model_test_score]

model_train_scores_df = pd.DataFrame(training_scores, index=algorithms, columns=['train_score']).reset_index()
model_train_scores_df.rename(columns={'index':'model'}, inplace=True)
model_train_scores_df.head()

model_test_scores_df = pd.DataFrame(testing_scores, index=algorithms, columns=['test_score']).reset_index()
model_test_scores_df.rename(columns={'index':'model'}, inplace=True)
model_test_scores_df.head()

model_scores = pd.merge(model_train_scores_df, model_test_scores_df, on='model')
model_scores.head()

model_scores.to_csv('resources/model_score.csv', index=False)

########################################
###         NEURAL NETWORK           ###
########################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create model and add layers
NN_model = Sequential()
NN_model.add(Dense(units=25, activation='relu', input_dim=X_top_feat_train_scaled.shape[1]))
NN_model.add(Dense(units=25, activation='relu'))
NN_model.add(Dense(units=2, activation='softmax'))

# Compile and fit the model
NN_model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
