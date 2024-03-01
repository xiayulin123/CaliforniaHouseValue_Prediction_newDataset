import pickle

from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('regModel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))
@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

def predict():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict_form', methods=['POST'])

def predict_form(): 
    data = [float(x) for x in request.form.values()]
    input = scalar.transform(np.array(data).reshape(1,-1))
    print(input)

    output = model.predict(input)[0]
    return render_template("home.html", prediction_text = "the value of the house is {}".format(output))

if __name__ == "__main__":
    app.run(debug = True)
