from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.decomposition import TruncatedSVD
from nolearn.dbn import DBN
from datetime import datetime

if sys.platform == 'darwin':
  os.chdir('/Users/Victor/Downloads/Kaggle-TalkingData/source')   # MacOS
else:
  os.chdir('C:\Users\Kun Wang\Documents\Kaggle\Kaggle-TalkingData\source')  # Windows

loaddir = '../processed_data/'
savedir = '../results/'

def load_sparse_csr(filename):
    """ load sparse matrix """
    print('Loading ', filename)
    loader = np.load(filename)
    return csr_matrix(( loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

    
def softmax_clf(X, y, weight=1.0, num_folds=5):
    """ softmax classifier """
    
    print('\nTraining softmax classifier...')
    clf = LogisticRegression(C=weight, multi_class='multinomial', solver='lbfgs')
    kf = StratifiedKFold(y, n_folds=num_folds, shuffle=True, random_state=0)
    scores = np.zeros((num_folds,1))
    cnt = 0
      
    for itrain, ival in kf:
        Xtrain, Xval = X[itrain,:], X[ival,:]
        ytrain, yval = y[itrain], y[ival]
        clf.fit(Xtrain, ytrain)
        ypred = clf.predict_proba(Xval)
        scores[cnt] = log_loss(yval, ypred)
        cnt += 1
        
    avg_score = np.mean(scores)
    print('C = {}, averag log loss = {}'.format(weight, avg_score))
    
    clf.fit(X, y)  # final training on all available data points
    return clf, avg_score
    

def bagging_clf(X, y, alg='rf', num_est=10, num_feat='auto'):
    """ Bagging methods: Random Forest or Extra Trees """
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=0)
    
    if alg == 'rf':
        print('\nTraining Random Forest Classifier...')
        clf = RandomForestClassifier(n_estimators=num_est, max_features=num_feat)
    else:
        print('\nTraining Extra Trees Classifier...')
        clf = ExtraTreesClassifier(n_estimators=num_est, max_features=num_feat)
    clf.fit(Xtrain, ytrain)
    
    ypred = clf.predict_proba(Xtest)
    score = log_loss(ytest, ypred)   # -sum(y_true*log(y_pred_prob)) : the lower the better
    print('num est = {}, max features = {}, Log loss = {}'.format(num_est, num_feat, score))
    
    return clf, score


def dbn_clf(X, y, hidden_sizes=[300], num_epochs=10):
    """ deep belief network """
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)
    output_categories = np.load(os.path.join(loaddir,'submit_col_name.npy'))

    print('Start training Neural Network...')

    dbn = DBN(
        [Xtrain.shape[1]] + hidden_sizes + [len(output_categories)],
        learn_rates = 0.3,
        learn_rate_decays = 0.9,
        epochs = num_epochs,
        verbose = 1)
    dbn.fit(Xtrain, ytrain)
    
    ypred = dbn.predict_proba(Xtest)
    score = log_loss(ytest, ypred)
    print('Log loss = {}'.format(score))

    return dbn, score


def make_submit(clf, pca=None):
    """ make file for submit """
    Xsubmit = load_sparse_csr(os.path.join(loaddir,'Xtest.npz'))
    if pca: Xsubmit = pca.transform(Xsubmit)    # PCA
    ysubmit = clf.predict_proba(Xsubmit)

    row_index = np.load(os.path.join(loaddir,'submit_row_index.npy'))
    col_name = np.load(os.path.join(loaddir,'submit_col_name.npy'))
    
    pred = pd.DataFrame(ysubmit, columns=col_name)
    # pred.head()
    
    pred.insert(0, 'device_id', row_index)    # insert at 0th col
    print(pred.head(3))

    print('\nSave submission...')
    savefile = 'submission-' + str(datetime.now()).split()[0] + '.csv'
    pred.to_csv(os.path.join(savedir,savefile),index=False)


def main_softmax():

    X = load_sparse_csr(os.path.join(loaddir,'Xtrain.npz'))
    y = np.load(os.path.join(loaddir,'ytrain.npy'))
    pca_switch, remain_ratio, pca = True, 0.05, None
    if pca_switch:
        pca = TruncatedSVD(n_components=int(remain_ratio*X.shape[1]))
        X = pca.fit_transform(X)
        print('Number of features:', X.shape[1])
    
    
    C = np.logspace(-4, 0, 10)
    # C = np.linspace(0.075, 0.085, 5)
    num_fold = 5

    best_clf, best_score, best_C = None, float('Inf'), 0

    for c in C:
        clf, score = softmax_clf(X, y, c, num_fold)
        if score < best_score:
            best_clf, best_score, best_C = clf, score, c

    print('\nBest score: {} at C = {}'.format(best_score, best_C))

    make_submit(best_clf, pca)
    print('\nDone')
    
    
def main_bagging():
    """ Too bad! """

    X = load_sparse_csr(os.path.join(loaddir,'Xtrain.npz'))
    y = np.load(os.path.join(loaddir,'ytrain.npy'))
    
    num_est = np.arange(10, 21, 10)
    max_feat = np.linspace(0.50, 1.50, 3) * np.sqrt(X.shape[1])
    alg = 'et'   # 'rf' or 'et'

    best_clf, best_score, best_num_est, best_max_feat = None, float('Inf'), 0, 0

    for est in num_est:
        for feat in max_feat:
            clf, score = bagging_clf(X, y, alg, int(est), int(feat))
            if score < best_score:
                best_clf, best_score, best_num_est, best_max_feat = clf, score, est, feat

    print('\nBest score: {} at num est = {}, max feat = {}'.format(best_score,\
    best_num_est, best_max_feat))

    make_submit(best_clf)
    print('\nDone')


def main_dbn():

    X = load_sparse_csr(os.path.join(loaddir,'Xtrain.npz'))
    y = np.load(os.path.join(loaddir,'ytrain.npy'))

    pca_switch, remain_ratio, pca = False, 0.03, None
    if pca_switch:
        pca = TruncatedSVD(n_components=int(remain_ratio*X.shape[1]))
        X = pca.fit_transform(X)
        print('Number of features after PCA:', X.shape[1])

    clf, score = dbn_clf(X, y, [500])

    make_submit(clf, pca)
    print('\nDone')
    

if __name__ == '__main__':
    # main_softmax()
#    main_bagging()
    main_dbn()

    
