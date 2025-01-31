from flask import Flask, render_template,request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model=pickle.load(open('model.pickle','rb'))
cols=['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[x for x in request.form.values()]
    final=np.array(features)
    data_unseen=pd.DataFrame([final],columns=cols)
    prediction=model.predict(data_unseen)
    output=round(prediction[0],2)
    return render_template('./index.html',prediction_text='CO2 Emission of the vehicle is {}'.format(output))


if __name__ == '__main__':
    app.run()

