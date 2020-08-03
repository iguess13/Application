from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

regressor = joblib.load("./linear_regression_model.pkl")

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        year = request.form['year']
        year = np.array([float(year)]).reshape(-1, 1)
        prediction = int(regressor.predict(year)[0][0])
        return render_template('index.html', result=prediction)
    else:
        value = np.array([2.1]).reshape(-1, 1)
        return render_template('index.html', result="A predire")


if __name__ == '__main__':
    app.run(debug=True)