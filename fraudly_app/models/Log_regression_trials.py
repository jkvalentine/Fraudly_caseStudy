# models!
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
# from sklearn.m import KFold
from sklearn.cross_validation import KFold
from statsmodels.api import Logit
from sklearn.metrics import accuracy_score, precision_score, recall_score

from data_pipeline import get_data, replace_delivery_nans

from smote import smote

df = pd.read_json('../fraud-detection-case-study/data/data.json')
X_train, X_test, y_train, y_test = fraudColumn_and_split(df)


X_train['delivery_method'] = X_train['delivery_method'].fillna(10.0)
X_train['email_com'] = (X_train.email_domain[-3:] == "com").astype(int)
X_train['email_org'] = (X_train.email_domain[-3:] == "org").astype(int)
X_train['email_edu'] = (X_train.email_domain[-3:] == "edu").astype(int)

X_train['num_payouts'] = X_train.previous_payouts.apply(lambda x: len(x))
X_train['venue_country!=country']= (X_train['country']!=X_train['venue_country']).astype(int)

def revenue(x):
	tot_revenue = 0
	for item in x:
		tot_revenue += item['cost']*item['quantity_sold']
	return tot_revenue

def expected_revenue(x):
	tot_revenue = 0
	for item in x:
		tot_revenue += item['cost']*item['quantity_total']
	return tot_revenue

def get_tix(ticket_types, value):
	total = 0
	for ticket in ticket_types:
		total += ticket[value]
	return total

X_train['actual_revenue'] = X_train.ticket_types.apply(revenue)
X_train['expected_revenue'] = X_train.ticket_types.apply(expected_revenue)
X_train['num_tix_total'] = X_train.ticket_types.apply(get_tix, args=("quantity_total",))
X_train['num_tix_sold_by_event'] = X_train.ticket_types.apply(get_tix, args=("quantity_sold",))	
	# previous tix sold (from previous_payouts)

# further division
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=.3)



# simplest model 'delivery_method',


models_dic = {'model0': ['fb_published','num_order'],
	'model1' : ['delivery_method','fb_published','num_order'],
	'model2' : ['delivery_method','fb_published','num_order','num_payouts'],
	'model3' : ['fb_published','num_payouts','venue_country!=country'],
	'model4' : ['fb_published','num_payouts','expected_revenue'],
	'model5' : []
	}


mod = 'model4'
# 
columns = models_dic[mod]
X_train = X_train[columns]
X_val = X_val[columns]

model = LogisticRegression()  # from sklearn
model.fit(X_train, y_train)
threshold = 0.15
predictions = model.predict_proba(X_val)[:, 1] > threshold

acc = accuracy_score(y_val,predictions)
presc = precision_score(y_val,predictions)
rec = recall_score(y_val,predictions)

print "for {} ==> Accuracy = {}, precision = {}, recall = {}".format(mod,acc,presc,rec)


# scores so far
# model0 ==> Accuracy = 0.691464629691, precision = 0.182600382409, recall = 0.720754716981
# model1 ==> Accuracy = 0.742942544005, precision = 0.212253829322, recall = 0.782258064516
# model2 ==> Accuracy = 0.819661242112, precision = 0.337713534823, recall = 0.868243243243
# model3 ==> Accuracy = 0.802059116573, precision = 0.283887468031, recall = 0.860465116279
# model4 ==> Accuracy = 0.543341082697, precision = 0.158650843223, recall = 0.900709219858
