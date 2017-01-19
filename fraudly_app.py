from flask import Flask, request, render_template, jsonify
import json
import requests
import socket
import time
import cPickle as pickle
import pandas as pd
from models import data_pipeline
from datetime import datetime

#Initialize pickle app
app = Flask(__name__)
PORT = 8080

#unpickle model
with open('gdbr.pickle') as f:
	model = pickle.load(f)

#unpickle tfidf
#with open('../tfidf.pkl') as g:
#	tfidf = pickle.load(g)


@app.route('/index')
def index():


	return render_template('index.html')

@app.route('/predict')
def predict():
	#Recieve heroku data
	raw_data = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
	data = raw_data.json()
	#Read batches of json files from NoSQL database

	#read json file into dataframe
	df = pd.Series(data, index = data.keys()).to_frame().T

	#transform data through data pipeline
	df_features = data_pipeline.feature_engineering(df)

	#do model prediction
	prediction = model.predict(df_features)
	if prediction == 1:
		message = "fraud"
	else:
		message = "not fraud"

	return render_template('/predict.html', prediction=prediction, id=data['object_id'] ,message=message)



@app.route('/about_us')
def about_us():
	return render_template('about_us.html')

if __name__ == '__main__':
    

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)