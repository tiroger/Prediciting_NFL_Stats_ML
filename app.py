

import os

import pandas as pd
import numpy as np

from flask import Flask, jsonify, render_template
#from flask_sqlalchemy import SQLAlchemy

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

@app.route("/predictions")
def make_predictions():
     """Returns prediction page"""
     return render_template("predictions.html")

if __name__ == "__main__":
     app.run()
