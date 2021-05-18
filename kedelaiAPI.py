import os 
import pickle
from typing import List
import numpy as np
from flask import Flask, request, jsonify


app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def status_api():
  return "API ON", 200

@app.route('/predict', methos=['POST'])
def predict():
  data = request.get_json(force=True)
  predict = model.predict(np.array([List(data.values())]))
  result = predict[0]

  answer = {'HEALTH': int(result)}
  return jsonify(answer)



if __name__ == "__main__":
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)