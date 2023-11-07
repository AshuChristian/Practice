from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

clf = joblib.load('model/linear_model.joblib')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    new_sample = np.array(data['sample'])
    prediction = clf.predict(new_sample)
    return jsonify({"result":prediction[0]})
    # classes = ["Positive","Negative"]
    # predicted_class = classes[prediction[0]]
    # return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000) 
