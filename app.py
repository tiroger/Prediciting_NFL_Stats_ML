

import os

import pandas as pd
import numpy as np

from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

# Route for collapsible speciess tree
@app.route("/model")
def show_model():
    return render_template("model.html")


@app.route("/data")
def show_data():
  	return nfl_data

      @TODO
      #Database stuff



if __name__ == "__main__":
     app.run()
