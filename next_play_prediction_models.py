# Next play Predictions

# Importing libraries and modules

# Data Wrangling
import pandas as pd
import numpy as np
import pandas_profiling


# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer #applies a power transformation to each feature to make the data more Gaussian-like.
from sklearn.model_selection import StratifiedKFold

##############################
#     DATA PROCESSING        #
##############################


# Opening dataset
play_by_play = pd.read_csv('resources/final_table_for_model_build.csv')
play_by_play.head()

# Keeping only play and run plays
play_by_play.play_type.value_counts()
# Plotting play types
x_pos = play_by_play.play_type.unique()
y_values = play_by_play.play_type.value_counts()
plt.ylabel('Play Type')
plt.xlabel('Number of Plays')
plt.title('Play Breakdown')
plt.barh(x_pos, y_values, color='SkyBlue')
plt.savefig('plots/play_breakdown_pre.png', dpi=600)
plt.tight_layout()
fig.show()

only_productive_plays = play_by_play[(play_by_play.play_type == 'pass') | (play_by_play.play_type == 'run')]
# Removing duplicates
only_productive_plays.drop_duplicates()
only_productive_plays.head()

# Plotting play types with only passes and runs
x_pos = only_productive_plays.play_type.unique()
y_values = only_productive_plays.play_type.value_counts()
plt.ylabel('Play Type')
plt.xlabel('Number of Plays')
plt.title('Play Breakdown')
plt.barh(x_pos, y_values, color='teal')
plt.tight_layout()
plt.savefig('plots/play_breakdown_post.png', dpi=600)
fig.show()


# Selecting features
X = only_productive_plays.drop(columns=['play_type', 'desc'])
X.head()
y = only_productive_plays.play_type
y.head()
print(X.shape)
print(y.shape)


########################################
###         FEATURE ENGINEERING      ###
########################################


feature_profile = X.profile_report()
feature_profile
feature_profile.to_file(output_file="feature_profile_preprocessing.html")

# Determining highly correlated feature_names
rejected_variables = feature_profile.get_rejected_variables(threshold=0.8)
rejected_variables

# Balancing the dataset
# importing SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter

# applying SMOTE to our data and checking the class counts
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_resampled.shape
y_resampled.shape


pt = PowerTransformer()
pt.fit(X_resampled)
X_resampled_transformed = pt.transform(X_resampled)
print(sorted(Counter(y_resampled).items()))

X_resampled_transformed.shape
y_resampled.shape

len(X_resampled_transformed)
len(y_resampled)



# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled_transformed, y_resampled, random_state=42)

X_train.shape
y_train.shape

# Saving training and testing datasets as csv for future use
from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save(outfile, X_train)
np.save(outfile, X_test)
np.save(outfile, y_train)
np.save(outfile, y_test)


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
LogReg_coefficients
LogReg_coefficients_array = LogReg_coefficients[0]
LogReg_coefficients_array
#Calculating odds ratio
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
RandFor_model = RandomForestClassifier(n_estimators=50)
RandFor_model.fit(X_train, y_train)

RandFor_model_train_score = RandFor_model.score(X_train, y_train)
RandFor_model_test_score = RandFor_model.score(X_test, y_test)
print(f"Training Data Score: {RandFor_model_train_score}")
print(f"Testing Data Score: {RandFor_model_test_score}")

#######################################
###  OPTMIZING RANDOM FOREST MODEL  ###
#######################################

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from yellowbrick.model_selection import CVScores

# Create a cross-validation strategy
cv = StratifiedKFold(n_splits=12, random_state=42)

# Instantiate the classification model and visualizer
model = RandomForestClassifier(n_estimators=50)
visualizer = CVScores(model, cv=cv, scoring='f1_weighted')

visualizer.fit(X_resampled, y_resampled)        # Fit the data to the visualizer
visualizer.show(outpath='plots/cross_validation.png', dpi=600)           # Finalize and render the figure


from sklearn.feature_selection import RFECV
rfc = RandomForestClassifier(random_state=100)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X_resampled, y_resampled)

print('Optimal number of features: {}'.format(rfecv.n_features_))

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.savefig('plots/RFECV.png', dpi=600)
plt.show()


print(np.where(rfecv.support_ == False)[0])

X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
X_resampled_selected = X.copy()
X_resampled_selected.to_csv('X_resampled_selected.csv', index=False)


dset = pd.DataFrame()
dset['attr'] = X.columns
dset['importance'] = rfecv.estimator_.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)
dset.head(8)

plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.savefig('plots/important_features.png', dpi=600)
plt.show()

#################################################
###  RUNNING MODELS ON OPTIMIZED FEATURE SET  ###
#################################################

X_optimal = only_productive_plays[dset.attr]
X_optimal.head()
y_optimal = only_productive_plays[['play_type']]
y_optimal.head()


X_optimal.shape
y_optimal.shape

X_optimal.head()
y_optimal.head()


# applying SMOTE to our data and checking the class counts
X_optimal_resampled, y_optimal_resampled = SMOTE().fit_resample(X_optimal, y_optimal)
X_optimal_resampled.shape
y_optimal_resampled.shape

# Saving training and testing datasets
X_optimal.to_csv('optimized_data/X_optimal.csv', index=False)
y_optimal.to_csv('optimized_data/y_optimal.csv', index=False)

pt = PowerTransformer()
pt.fit(X_optimal_resampled)
X_optimal_resampled_transformed = pt.transform(X_optimal_resampled)
print(sorted(Counter(y_resampled).items()))


#Splitting dataset
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_optimal_resampled_transformed, y_optimal_resampled, random_state=42)

X_optimal_resampled


RandFor = RandomForestClassifier(n_estimators=300, min_samples_split=4, class_weight='balanced')
RandFor.fit(X_new_train, y_new_train)

RandFor_model_train_score = RandFor.score(X_new_train, y_new_train)
RandFor_model_test_score = RandFor.score(X_new_test, y_new_test)
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
model_scores.to_json('resources/model_score.json', orient='records')

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
