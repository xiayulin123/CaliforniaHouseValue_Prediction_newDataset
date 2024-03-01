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

@app.route('/predict', method=['POST'])

def predict():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(list(data.values()).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug = True)
