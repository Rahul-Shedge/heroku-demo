from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# import pandas as pd
# import datetime

app = Flask(__name__,template_folder='template')
Load_model = load_model("Monthly_Model.h5")
Load_model1 = load_model("Monthly_Model1.h5")
const port=process.env.port || '8080'

app=express();
app.set("port",PORT);

Scaler = pickle.load(open('MinmaxScaler.pkl', 'rb'))
Scaler1 = pickle.load(open('MinmaxScaler1.pkl', 'rb'))

@app.route("/", methods=["GET", "POST"])
def homey():
    return render_template('entry.html')
@app.route("/index", methods=["GET","POST"])
def index():
    return render_template('index.html')

@app.route('/method', methods=['POST'])
def predict():
    Value = request.form["Close Price"]
    scaled = Scaler.transform(np.array(Value).reshape(-1, 1))
    prediction = Load_model.predict(scaled.reshape(scaled.shape[0], 1, scaled.shape[1]))
    Result = Scaler.inverse_transform(prediction)[0][0]
    return render_template("result.html", result=Result)

@app.route("/index1", methods=["GET","POST"])
def index1():
    return render_template('index1.html')


@app.route('/method1', methods=['POST'])
def predictt():
    int_features=[int(x) for x in request.form.values()]
    scaled1 = Scaler1.transform(np.array(int_features).reshape(-1, 1))
    prediction1 = Load_model1.predict(scaled1.reshape(scaled1.shape[0], 1, scaled1.shape[1]))
    Result1 = Scaler1.inverse_transform(prediction1)[0][0]
    return render_template("result1.html", result1=Result1)


if __name__ == "__main__":
    app.run(debug=True)
