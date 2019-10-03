

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
    """Returns the homepage"""
    return render_template("contacts.html")

# @app.route("/data")
# def show_data():
#   	return nfl_data

#       @TODO
#       #Database stuff



if __name__ == "__main__":
     app.run()
