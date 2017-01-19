import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import data_pipeline
import cPickle as pickle
from data_pipeline import get_data, feature_engineering, get_tix, scale_data
import seaborn as sb

def random_forrest_class_balence(scaler_train, scaler_test):

    #random_forrest
    rf =  RandomForestClassifier(n_estimators=3, oob_score=True, class_weight={1:9})
    rf.fit(scaler_train, y1_train)
    scaler_test_predict = rf.predict(scaler_test)

    rs = recall_score(scaler_test_predict, y1_test)
    ps = precision_score(scaler_test_predict, y1_test)
    f1 = f1_score(y1_test, scaler_test_predict)
    cm_rf =  confusion_matrix(scaler_test_predict, y1_test)
    print 'random forrest'
    print "recall_score", rs
    print 'precision_score', ps
    print " accuracy score:", rf.score(scaler_test, y1_test)
    print "f1 score", f1
    feature_importances = np.argsort(rf.feature_importances_)
    print " top 10 features:", list(x_feature_train.columns[feature_importances[-1:-10:-1]])
    print "confusion_matrix:", confusion_matrix(scaler_test_predict, y1_test)
    print '------------------------------------------------------'
    filename = 'rf.pickle'
    save_to_pickle(filename, rf)

    return rf, cm_rf
    #recall_score 0.944134078212
    #precision_score 0.871134020619
    #accuracy score: 0.982561036373
    #top 10 features: ['has_previous_payouts', u'sale_duration2', 'num_tix_sold_by_event', u'body_length', u'num_payouts', u'approx_payout_date', 'venue_outside_user_country', u'num_order', 'num_tix_total']
def gdbr(scaler_train, scaler_test):
    gdbr = GradientBoostingClassifier(learning_rate=0.03,
                                 n_estimators=150, random_state=1)
    # weights = y1_train
    # weights.loc[weights==1] = 20
    # weights.loc[weights==0] = 80

    gdbr.fit(scaler_train, y1_train)
    scaler_test_predict = gdbr.predict(scaler_test)

    rs = recall_score(scaler_test_predict, y1_test)
    ps = precision_score(scaler_test_predict, y1_test)
    f1 = f1_score(y1_test, scaler_test_predict)
    cm = confusion_matrix(scaler_test_predict, y1_test)   #F1 = 2 * (precision * recall) / (precision + recall)
    print 'gdbr'
    print "recall_score", rs
    print 'precision_score', ps
    print "f1 score", f1
    print " accuracy score:", gdbr.score(scaler_test, y1_test)
    print "confusion_matrix:", confusion_matrix(scaler_test_predict, y1_test)

    filename = 'gdbr.pickle'
    save_to_pickle(filename, gdbr)
    #recall_score 0.958333333333
    #precision_score 0.879781420765
    #accuracy score: 0.985550572995
    return gdbr, cm

def cross_val(estimator, train_x, train_y):
    # n_jobs=-1 uses all the cores on your machine
    mse = cross_val_score(estimator, train_x, train_y,
                           scoring='mean_squared_error',
                           cv=5) * -1
    r2 = cross_val_score(estimator, train_x, train_y,
                           scoring='r2', cv=5)

    mean_mse = mse.mean()
    mean_r2 = r2.mean()
    params = estimator.get_params()
    name = estimator.__class__.__name__
    print '%s Train CV | MSE: %.3f | R2: %.3f' % (name, mean_mse, mean_r2)
    return mean_mse, mean_r2

def plot_feature_importance():
    indices = np.argsort(gdbr.feature_importances_)
    # plot as bar chart
    figure = plt.figure(figsize=(10,7))
    plt.barh(np.arange(len(columns)), gdbr.feature_importances_[indices],
             align='center', alpha=.5)
    plt.yticks(np.arange(len(columns)), np.array(columns)[indices], fontsize=14)
    plt.xticks(fontsize=14)
    _ = plt.xlabel('Relative importance', fontsize=18)
    plt.show()

def save_to_pickle(filename, model):
    with open(filename, 'w') as f:
        pickle.dump(model, f)


def standard_confusion_matrix(y_true, y_predict):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_predict)
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit_matrix, probabilities, y_true):
    thresholds = sorted(probabilities)
    thresholds.append(1.0)
    profits = []
    for threshold in thresholds:
        y_predict = probabilities >= threshold
        confusion_mat = standard_confusion_matrix(y_true, y_predict)
        profit = np.sum(confusion_mat * cost_benefit_matrix) / float(len(y_true))
        profits.append(profit)
    return thresholds, profits

def run_profit_curve(model, costbenefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds, profits = profit_curve(costbenefit, probabilities, y_test)
    return thresholds, profits

def plot_profit_models(models, costbenefit, X_train, X_test, y_train, y_test):
    percentages = np.linspace(0, 100, len(y_test) + 1)
    for model in models:
        thresholds, profits = run_profit_curve(model,
                                               costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)
        plt.plot(percentages, profits, label=model.__class__.__name__)
    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='upper left')
    plt.savefig('profit_curve.png')
    plt.show()

def find_best_threshold(models, costbenefit, X_train, X_test, y_train, y_test):
    max_model = None
    max_threshold = None
    max_profit = None
    for model in models:
        thresholds, profits = run_profit_curve(model, costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model.__class__.__name__
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    X1_train, x1_test, y1_train, y1_test = train_test_split(X_train, y_train, test_size=.2)
    columns = ['has_previous_payouts', 'gts_is_0', 'gts_less_10', 'gts_less_25', 'venue_outside_user_country', 'num_tix_total', 'num_tix_sold_by_event', 'num_payouts', 'email_gmail', 'email_yahoo', 'email_hotmail','email_aol','email_com', 'email_org', 'email_edu','approx_payout_date', 'sale_duration2', 'num_order', 'body_length', 'high_fraud_country', 'exclamation_points']


    x_feature_train = feature_engineering(X1_train)
    x_feature_test = feature_engineering(x1_test)

    scaler_train, scaler_test = scale_data(x_feature_train, x_feature_test)

    rf2, cm_rf = random_forrest_class_balence(scaler_train, scaler_test)
    gdbr, cm_gdbr = gdbr(scaler_train, scaler_test)

    cross_val(rf2, scaler_train, y1_train)
    cross_val(gdbr, scaler_train, y1_train)

    plot_feature_importance()

    costbenefit = np.array([[100, 0], [-200, 0]])
    models = [rf2, gdbr]
    plot_profit_models(models, costbenefit, scaler_train, scaler_test, y1_train, y1_test)
    print find_best_threshold(models, costbenefit,
                             scaler_train, scaler_test, y1_train, y1_test)
