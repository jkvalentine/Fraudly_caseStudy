from flask import Flask, request, render_template
import json
import requests
import socket
import time
import pandas as pd
from models import data_pipeline
from datetime import datetime

#Initialize pickle app
app = Flask(__name__)
PORT = 8080

#unpickle model
with open('../best_model.pkl') as f:
	model = pickle.load(f)

#unpickle tfidf
with open('../tfidf.pkl') as g:
	tfidf = pickle.load(g)


@app.route('/index')
def index():


	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	#Recieve heroku data
	data = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')

	#Read batches of json files from NoSQL database

	#read json file into dataframe
	df = pd.read_json(data)

	#transform data through data pipeline



	#do model prediction
	model.predict()

	return 



@app.route('/about_us')
def about_us():
	return render_template('about_us.html')

if __name__ == '__main__':
    

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)