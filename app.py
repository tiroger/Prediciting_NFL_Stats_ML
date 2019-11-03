

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
    # to_predict = to_predict_list
    to_predict = np.array(to_predict_list).reshape(1, 8)
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)
    print(to_predict)
    testvalue = [[19.0, 3, 1.0, 10, 3.0, 3.0, 35, 74]]
    print(testvalue)

    # loaded_model = pickle.load(open("model.pkl","rb"))
    # result = loaded_model.predict(to_predict)
    # return result[0]

    # input = []
    # for value in inputValue.values():
    #     input.append(value)
    #     input_df = pd.DataFrame(input)
    #     input_df.rename(columns={0: 'input_value'}, inplace=True)
    #     to_predict = input_df.transpose()  # or df1.transpose()
    # loaded_model = joblib.load('resources/next_play_predictor_LogReg.pkl')
    loaded_model = pickle.load(
        open("resources/next_play_predictor_LogReg.pkl", "rb"))
    # result = loaded_model.predict(to_predict)
    result = loaded_model.predict(testvalue)
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


@app.route("/demoday")
def demo_day():
    """Returns the model building page"""
    return render_template("demoday.html")


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
    prediction = "pass"
    if request.method == 'POST':
        print("RECEIVED ROUTE REQUEST!!!!!!!")
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))

        print(to_predict_list)
        formData = request.form.to_dict()
        print(formData)

        temp = int(request.form['temp'])
        humidity = int(request.form['humidity'])
        # temp = int(request.form['temp'])
        # temp = int(request.form['temp'])
        # temp = int(request.form['temp'])
        # temp = int(request.form['temp'])
        # temp = int(request.form['temp'])
        # temp = int(request.form['temp'])

        prediction = predictPlay(to_predict_list)
        print(prediction)

    return render_template("predictions.html", prediction=prediction)


# @app.route('/result', methods = ['POST'])
# def result():
#     if request.method == 'POST':

#         # any transforms
#         modelValue = request.form.to_dict()
#         modelValue = list(modelValue.values())
#         modelValue = list(map(int, modelValue))

#         prediction = predictPlay(modelValue)

#         return render_template("predictions.html",prediction=prediction)


if __name__ == "__main__":
    app.run()
