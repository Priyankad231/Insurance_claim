import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Load the model once at startup for efficiency
with open("model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

app = Flask(__name__)

def ValuePredictor(new_l):
    # Reshape input for prediction
    to_predict = np.array(new_l).reshape(1, 7)
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def home_():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        new_l = []
        for x in to_predict_list:
            # Convert categorical values to numerical
            if x.lower() == 'male':
                new_l.append(1)
            elif x.lower() == 'female':
                new_l.append(0)
            elif x.lower() == 'yes':
                new_l.append(1)
            elif x.lower() == 'no':
                new_l.append(0)
            else:
                new_l.append(x)

        # Convert all inputs to float
        try:
            new_l = list(map(float, new_l))
        except ValueError:
            return render_template("result.html", prediction="Invalid input. Please enter valid numbers.")

        # Make prediction
        result = ValuePredictor(new_l)
        if int(result) == 1:
            prediction = 'Yes, the person can claim insurance.'
        else:
            prediction = 'No, the person will not claim insurance.'

        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
