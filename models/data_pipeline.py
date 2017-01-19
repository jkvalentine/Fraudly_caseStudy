import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.feature_extraction import text
from numpy.linalg import lstsq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from bs4 import BeautifulSoup
from collections import Counter
import string


def get_data():
	df = pd.read_json('data/data.json')

	#Create binary fraud column
	df['fraud'] = 0
	df.loc[df['acct_type'] == 'fraudster_event', 'fraud'] = 1
	df.loc[df['acct_type'] == 'fraudster', 'fraud'] = 1
	df.loc[df['acct_type'] == 'fraudster_att', 'fraud'] = 1
	#Check
	# print "Should be 1293, it is... "+ sum(df['fraud'])

	#Train, test, split
	y = df['fraud']
	X = df[[         u'approx_payout_date',        u'body_length',
		     u'channels',            u'country',           u'currency',
	      u'delivery_method',        u'description',       u'email_domain',
		u'event_created',          u'event_end',    u'event_published',
		  u'event_start',       u'fb_published',                u'gts',
		u'has_analytics',         u'has_header',           u'has_logo',
		       u'listed',               u'name',        u'name_length',
		    u'num_order',        u'num_payouts',          u'object_id',
		     u'org_desc',       u'org_facebook',           u'org_name',
		  u'org_twitter',         u'payee_name',        u'payout_type',
	     u'previous_payouts',      u'sale_duration',     u'sale_duration2',
		     u'show_map',       u'ticket_types',           u'user_age',
		 u'user_created',          u'user_type',      u'venue_address',
		u'venue_country',     u'venue_latitude',    u'venue_longitude',
		   u'venue_name',        u'venue_state',              ]]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)
	return X_train, X_test, y_train, y_test


def feature_engineering(df):

	# are there any previous payouts?

	df['has_previous_payouts'] = df.previous_payouts.apply(lambda x: int(x == []))

	# gts values -- binned

	df['gts_is_0'] = df.gts.apply(lambda x: int(x == 0))
	df['gts_less_10'] = df.gts.apply(lambda x: int(0 < x < 10))
	df['gts_less_25'] = df.gts.apply(lambda x: int(10 < x < 25))

	# user country != venue country

	country_mismatch = df.venue_country != df.country
	df['venue_outside_user_country'] = country_mismatch.astype(int)

	#country dummies

	#num of tix for sale (from ticket types)
	df['num_tix_total'] = df.ticket_types.apply(get_tix, args=("quantity_total",))
	# num of tix sold (from ticket types)
	df['num_tix_sold_by_event'] = df.ticket_types.apply(get_tix, args=("quantity_sold",))	
	# previous tix sold (from previous_payouts)
	df['num_payouts'] = df.previous_payouts.apply(lambda x: len(x))	

	#emails:
	df['email_gmail'] = (df.email_domain == "gmail.com").astype(int)
	df['email_yahoo'] = (df.email_domain == "yahoo.com").astype(int)
	df['email_hotmail'] = (df.email_domain == "hotmail.com").astype(int)
	df['email_aol'] = (df.email_domain == "aol.com").astype(int)
	df['email_com'] = (df.email_domain.apply(lambda x: x[-3:]) == "com").astype(int)
	df['email_org'] = (df.email_domain.apply(lambda x: x[-3:]) == "org").astype(int)
	df['email_edu'] = (df.email_domain.apply(lambda x: x[-3:]) == "edu").astype(int)

	#fraudy countries
	# fraud_one_sd_above = np.mean(df['fraud']) + np.std(df['fraud'])
	# fraud_bools = df.groupby('country').mean()['fraud'] > fraud_one_sd_above
	# high_fraud=df.groupby('country').mean()[fraud_bools]
	# high_fraud_countries = high_fraud.index
	# df['high_fraud_country'] = df.country.apply(lambda x: x in high_fraud_countries).astype(int)
	fraudy_countries = [u'A1', u'AR', u'BG', u'CH', u'CI', u'CM', u'CN', u'CO', u'CZ', u'DE',
      u'DK', u'DZ', u'FI', u'HR', u'ID', u'IL', u'JE', u'JM', u'KH', u'MA',
      u'MY', u'NA', u'NG', u'PH', u'PK', u'PR', u'PS', u'QA', u'RU', u'TR',
      u'VN']
    df['high_fraud_country'] = df.country.apply(lambda x: 1 if x in fraudy_countries else 0)

	# get number of exclamation marks
	df['exclamation_points'] = df['description'].apply(count_bangs)

	# get proportion of caps
	df['caps_proportion'] = df['description'].apply(caps_prop)

	# make columns according to latent topics
	df = topic_dummies(df)
	return df

def count_bangs(description_string):
   char_count = Counter(description_string)
   return char_count['!']

def caps_prop(description_string):
   if description_string:
       return len(filter(lambda x: x in string.uppercase, description_string))/len(description_string)

def scale_data(x_train, x_test):
	scaler = preprocessing.StandardScaler()
	scaler.fit(x_train)
	scaler_train = scaler.transform(x_train)
	scaler_test = scaler.transform(x_test)

	return scaler_train, scaler_test

def create_X_and_y(df):
	df = feature_engineering(df)
	y = df['fraud']
	X = df.drop('fraud')
	return X, y

def test_train_split(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
	return X_train, X_test, y_train, y_test

def replace_delivery_nans(df): 
	df['delivery_method'].ix[df['delivery_method'].isnull()] = 10.0
	return df

def get_tix(ticket_types, value):
	total = 0
	for ticket in ticket_types:
		total += ticket[value]
	return total

def topic_dummies(df):

    #CLEAN HTML FUNCTION
    def get_text(cell):
        return BeautifulSoup(cell, 'html.parser').get_text()

    #Parse descriptions using html function above:
    df['description'] = df['description'].apply(get_text)
    df['org_desc'] = df['org_desc'].apply(get_text)
    clean = df['description']

    #All the parameters for the topic modeling.
    n_samples = len(clean)
    n_features = 500
    n_topics = 9
    n_top_words = 30

    my_additional_stopwords = ["la", "et", "en", "le", "les", "des", 'january', 'february',
                           'march', 'april', 'may', 'june', 'july', 'august', 'september',
                           'october', 'november', 'december', 'friday', 'thursday', 'saturday']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stopwords)


    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words=stop_words)
    tfidf = tfidf_vectorizer.fit_transform(clean)

    # Fit the NMF model
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    #Leave this turned off unless you want to print.
    #tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    #print_top_words(nmf, tfidf_feature_names, n_top_words)

    '''
    #Assign topics to descriptions:
    #These are from the full data.  Do NOT use these descriptions on any subset, as they will not match.
    topic_dict = {0:'dinner_party', 1:'educational', 2:'social_networks', 3:'logistics', 4: 'business', 5:'university',
                  6:'club_logistics', 7:'workshop', 8:'club_content'}
    '''
    topic_dict = {0:'topic1', 1:'topic2', 2:'topic3', 3:'topic4', 4: 'topic5', 5:'topic6',
                  6:'topic7', 7:'topic8', 8:'topic9'}


    W = nmf.transform(tfidf)
    df['topic_index'] = np.argmax(W, axis=1)
    df['topic_index'] = df['topic_index'].replace(topic_dict)

    ###Create dummy variables to insert into model
    topic_dummies = pd.get_dummies(df['topic_index']).rename(columns = lambda x: 'topic_'+str(x))
    df = pd.concat([df,topic_dummies],axis=1)
    return df

def smote(X, y, target, k=None):
	"""
	INPUT:
	X, y - your data
	target - the percentage of positive class 
	     observations in the output
	k - k in k nearest neighbors
	OUTPUT:
	X_oversampled, y_oversampled - oversampled data
	`smote` generates new observations from the positive (minority) class:
	For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
	# """
	# if target <= sum(y)/float(len(y)):
	# 	return X, y
	# if k is None:
	# k = len(X)**.5
	# # fit kNN model
	# knn = KNeighborsClassifier(n_neighbors=k)
	# knn.fit(X[y==1], y[y==1])
	# neighbors = knn.kneighbors()[0]
	# positive_observations = X[y==1]
	# # determine how many new positive observations to generate
	# positive_count = sum(y)
	# negative_count = len(y) - positive_count
	# target_positive_count = target*negative_count / (1. - target)
	# target_positive_count = int(round(target_positive_count))
	# number_of_new_observations = target_positive_count - positive_count
	# # generate synthetic observations
	# synthetic_observations = np.empty((0, X.shape[1]))
	# while len(synthetic_observations) < number_of_new_observations:
	# obs_index = np.random.randint(len(positive_observations))
	# observation = positive_observations[obs_index]
	# neighbor_index = np.random.choice(neighbors[obs_index])
	# neighbor = X[neighbor_index]
	# obs_weights = np.random.random(len(neighbor))
	# neighbor_weights = 1 - obs_weights
	# new_observation = obs_weights*observation + neighbor_weights*neighbor
	# synthetic_observations = np.vstack((synthetic_observations, new_observation))

	# X_smoted = np.vstack((X, synthetic_observations))
	# y_smoted = np.concatenate((y, [1]*len(synthetic_observations)))

	# return X_smoted, y_smoted
	pass


def smote2(X, y, target, k=None):
	"""
	INPUT:
	X, y - your data
	target - the percentage of positive class 
	     observations in the output
	k - k in k nearest neighbors
	OUTPUT:
	X_oversampled, y_oversampled - oversampled data
	`smote` generates new observations from the positive (minority) class:
	For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
	"""
	'''

	y_zeros = y[y==0]
	X_zeros = X[y==0]

	y_ones = y[y==1]
	X_ones = X[y==1]

	if len(y_ones) > len(y_zeros):
	y_minority = y_zeros
	X_minority = X_zeros
	else:
	y_minority = y_ones
	X_minority = X_ones

	# fit a KNN model    
	# This has to be called on the minority bunch only!!!!!    
	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_minority)
	distances, indices = nbrs.kneighbors(X_minority)

	# determine how many new positive observations to generate    
	target = float(target)    
	N_new_data = (len(y)*target - len(y_minority))/(1-target)

	# adding to the zeros
	ind_new = np.random.randint(0,len(y_minority),N_new_data)


	# generate synthetic observations

	y_synth = np.zeros(len(N_new_data))
	X_synth = []        
	for value in ind_new:
	r = np.random(0, k)           
	neighbor_index = indices[value, r]
	distances = np.random.random(0, 1, len(X_minority.columns))            
	new_point = X_minority[value] + distances*(X_minority[value]-X_minority[neighbor_index])            
	X_synth.append = new_point

	# combine synthetic observations with original observations
	X_smoted = np.concatenate((X_ones, X_zeros, X_synth),axis=1)
	y_smoted = np.concatenate((y_ones, y_zeros, y_synth),axis=1)

	return X_smoted, y_smoted
	'''
	pass
