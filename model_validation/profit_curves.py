# profit curves
import numpy as np 
import cPickle as pickle
import pandas as pd

# confusion matrix:

def conf_matrix(y_true,y_predict):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)    
    tp =  sum(y_predict & y_true)   
    fp = sum(y_predict & np.logical_not(y_true))    
    fn = sum(np.logical_not(y_predict) & (y_true))   
    tn = sum(np.logical_not(y_predict) & np.logical_not(y_true))   
    return np.array([[tp, fp],[fn,tn]])

def profit_curve(cb, predict_probas, labels):
    profits = []
    thresholds = np.unique(predict_probas)
    Perc_Pred_Pos = []
    for threshold in thresholds:
        profits.append(float((cb*conf_matrix(labels, predict_probas>threshold)).sum())/len(predict_probas))
        Perc_Pred_Pos.append((predict_probas>threshold).sum()/len(predict_probas))
    return profits[::-1]

def plot_profit_curve(model, label, costbenefit, X_test, y_test):
    
    # model should be fitted already (maybe from a pickle)
    model_instance = model
    # model_instance.fit(X_train,y_train)  
    predict_probas = model_instance.predict_proba(X_test)
    profits = profit_curve(costbenefit, predict_probas[:,1], y_test)
    
    percentages = np.arange(0, 100, 100. / len(profits))
    
    plt.plot(percentages, profits, label=label)
    plt.title("Profit Curve Per Person")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='lower right')    
    plt.show()
#     return percentages, profits

if __name__ == '__main__':
	tp = 11
	fp = -23
	fn = 0
	tn = 3
	# cost-benefit matrix
	cb_matrix = np.array([[tp, fp],[fn,tn]])
	with open('data/model.pkl') as f:
	model = pickle.load(f)
	plot_profit_curve(model,'model',cb_matrix, X_test, y_test)
	
	
