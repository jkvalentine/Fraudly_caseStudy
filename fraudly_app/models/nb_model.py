import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import feature_engineering,\
test_train_split, create_X_and_y

def read_data(filename):
	df = read_json(filename)

if __name__ if __name__ == '__main__':
	#need to figure out how to generalize load data
	df = read_data(filename)

	#run data through pipeline for features and transformations
	df_features = feature_engineering(df)
	
	#Create X and y for splits, models
	X, y = create_X_and_y(df_features)

	#Create test and training sets
	X_train, X_val, y_train, y_val = test_train_split(X, y)
	
	#Instantiate scaler, scale data for Naive Bayes model
	min_max = MinMaxScaler()
	min_max.fit(X_train)
	X_scaled = min_max.transform(X_train)

	#Instantiate, fit Naive Bayes model
	nb = MultinomialNB()
	nb.fit(scaled_data, y_train)

	#Transform X_validation data for NB prediction
	X_val_scaled = min_max.transform(X_val)

	#Make prediction for validation data
	y_predict = nb.predict(X_val_scaled)
	print "y_predict is {}".format(y_predict)

	#Calculate precision, accuracy, recall
	tp = int(sum(y_val * y_predict))
	fn = int(sum((1-y_val) * y_predict))
	fp = int(sum(y_val * (1-y_predict)))

	recall = float(tp)/(fn+tp)
	precision = float(tp)/(tp+fp)

	print "recall is {}".format(recall)
	print "precision is {}".format(precision)
	print "accuracy is {}".format(nb.score(X_val_scaled, y_val))



	