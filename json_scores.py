# Returns the model scores as json object
from flask import jsonify
import pandas as pd

def model_scores() :
    json_ = request.json
    new = pd.read_csv('models.csv')
    json_vector = new.transform(json_)
    query = pd.DataFrame(json_vector)
    prediction = regr.predict(query)
    data = {'prediction': list({{prediction}})}

    return jsonify(data)
