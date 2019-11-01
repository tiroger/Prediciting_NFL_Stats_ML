

import os

import pandas as pd
import numpy as np

from flask import Flask, jsonify, render_template
#from flask_sqlalchemy import SQLAlchemy

# To Return json of scores
import json

app = Flask(__name__)


@app.route("/")
def index():
    """Returns the homepage"""
    return render_template("index.html")

@app.route("/contact-info")
def show_contact():
    """Returns the contact info page"""
    return render_template("contacts.html")

@app.route("/visualizations")
def show_visualizations():
    """Returns the visualizations"""
    return render_template("visualizations.html")

@app.route("/model-build")
def show_model():
    """Returns the model building page"""
    return render_template("model-build.html")

@app.route("/feature-profile")
def show_profile():
    """Returns the feature profile page"""
    return render_template("feature_profile.html")

# @app.route("/box-plot")
# def show_box():
#     """Returns the feature profile page"""
#     return render_template("figure_68.html")


@app.route("/scores")
def model_scores():
    # Loading the JSON file's data into file_data
    # every time a request is made to this endpoint
    with open('resources/model_score.json', 'r') as jsonfile:
        file_data = json.loads(jsonfile.read())
    # We can then find the data for the requested date and send it back as json
    return json.dumps(file_data)

@app.route("/predictions")
def make_predictions():
     """Returns prediction page"""
     return render_template("predictions.html")


def predictPlay(inputValue):
    input = []
    for value in inputvalue.values():
          input.append(value)

    input_df = pd.DataFrame(input)
    input_df.rename(columns = {0:'input_value'}, inplace=True)
    to_predict = input_df.transpose() # or df1.transpose()
    loaded_model = joblib.load('resources/next_play_predictor_LogReg.pkl')
     # loaded_model = pickle.load(open("/resoureces/next_play_predictor_LogReg.pkl","rb"))
    result = loaded_model.predict(to_predict)

    return result

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':

        # any transforms
        modelValue = request.form.to_dict()
        modelValue = list(modelValue.values())
        modelValue = list(map(int, modelValue))

        prediction = predictPlay(modelValue)

        return render_template("predictions.html",prediction=prediction)


# @app.route("/score")
# def score():
#     print(request.args)
#     if(request.args):
#         x_input, predictions = make_prediction(request.args(modelValue))
#         print(x_input)
#         return render_template('predictions.html', response=prediction)
#     else:
#         #For first load, request.args will be an empty ImmutableDict
#         # type. If this is the case we need to pass an empty string
#         # into make_prediction function so no errors are thrown.
#         x_input, predictions = make_prediction('')
#         return render_template('predictions.html', response=null)

if __name__ == "__main__":
     app.run()
