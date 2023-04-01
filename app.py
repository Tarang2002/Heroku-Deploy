from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('D:\College\sem 6\Mini Project\model\model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    A1_Score = request.form.get('A1_Score')
    A2_Score = request.form.get('A2_Score')
    A3_Score = request.form.get('A3_Score')
    A4_Score = request.form.get('A4_Score')
    A5_Score = request.form.get('A5_Score')
    A6_Score = request.form.get('A6_Score')
    A7_Score = request.form.get('A7_Score')
    A8_Score = request.form.get('A8_Score')
    A9_Score = request.form.get('A9_Score')
    A10_Score = request.form.get('A10_Score')
    age	= request.form.get('age')
    ethnicity = request.form.get('ethnicity')
    contry_of_res = request.form.get('contry_of_res')
    Jaundice = request.form.get('Jaundice')
    Austim = request.form.get('Austim')
    Female = request.form.get('Female')
    Male = request.form.get('Male')

    input_query = np.array([[A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score, age, ethnicity, contry_of_res, Jaundice, Austim, Female, Male]])

    result = model.predict(input_query)[0]

    return jsonify({'ASD':str(result)})

if __name__ == '__main__':
    app.run(debug=True)