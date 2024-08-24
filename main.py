from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

data= pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('indec.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    total_sqft = float(request.form.get('total'))
    bath = float(request.form.get('bath'))
    input = pd.DataFrame([[location,total_sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0] * 1e5
    return str(np.round(prediction,2))
if __name__ == '__main__':
    app.run(debug=True)