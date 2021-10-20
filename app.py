from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        sepal_len = request.form.get('sepal_length')
        sepal_wid = request.form.get('sepal_width')
        petal_len = request.form.get('petal_length')
        petal_wid = request.form.get('petal_width')

        try:
            prediction = preprocessDataAndPredict(
                sepal_len, sepal_wid, petal_len, petal_wid)
            return render_template('predict.html', prediction=prediction)
        except ValueError:
            return "Pls enter valid values"
        pass
    pass


def preprocessDataAndPredict(sepal_len, sepal_wid, petal_len, petal_wid):
    test_data = [sepal_len, sepal_wid, petal_len, petal_wid]
    test_data = np.array(test_data)
    test_data = test_data.reshape(1, -1)
    file = open("iris_model_rf.pkl", "rb")
    trained_model = joblib.load(file)
    prediction = trained_model.predict(test_data)
    return prediction

    pass


if __name__ == '__main__':
    app.run(debug=True)
