

import os
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle

from flask import Flask, jsonify, render_template, request
# from flask_sqlalchemy import SQLAlchemy

# To Return json of scores
import json

app = Flask(__name__)


def predictPlay(to_predict_list):
    # reshape to prep data for prediction
    to_predict = np.array(to_predict_list).reshape(1, 7)
    # scale input values to trained models
    pt_scaler = joblib.load('resources/pt_scaler.sav')
    to_predict_final = pt_scaler.transform(to_predict)

    # load spedicified model for use and make prediction
    loaded_model = joblib.load("resources/logreg_model.sav")
    result = loaded_model.predict(to_predict_final)
    return result


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


@app.route("/feature-profile")
def feature_profile():
    """Returns the model building page"""
    # return render_template("feature_profile.html")
    # Below is test for rendering a page with layout
    return render_template("feature_prof.html")


@app.route("/model-build")
def show_model():
    """Returns the model building page"""
    return render_template("model-build.html")


@app.route("/scores")
def model_scores():
    # Loading the JSON file's data into file_data
    # every time a request is made to this endpoint
    with open('resources/model_score.json', 'r') as jsonfile:
        file_data = json.loads(jsonfile.read())
    # We can then find the data for the requested date and send it back as json
    return json.dumps(file_data)


@app.route("/predictions", methods=["GET", "POST"])
def make_predictions():
    """Returns prediction page"""

    return render_template("predictions.html")


@app.route('/results', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))

        # call the function to make our prediction
        prediction = predictPlay(to_predict_list)

        if prediction == "run":
            result = "run"
        elif prediction == "pass":
            result = "pass"
        elif prediction == "field_goal":
            result = "field goal"
        else:
            result = "punt"
        prediction = result.upper()
        return jsonify(prediction)


if __name__ == "__main__":
    app.run()
