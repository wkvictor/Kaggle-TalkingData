from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
#import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
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

    
def softmax_clf(weight, X, y):
    """ softmax classifier """
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=5183)
    
    print('\nTraining softmax classifier...')
    clf = LogisticRegression(C=weight, multi_class='multinomial', solver='lbfgs')
    clf.fit(Xtrain, ytrain)
    
    ypred = clf.predict_proba(Xtest)
    score = log_loss(ytest, ypred)   # -sum(y_true*log(y_pred_prob)) : the lower the better
    print('C = {}, Log loss = {}'.format(weight, score))
    
    clf.fit(X, y)  # final training on all available data points
    return clf, score


def make_submit(clf):
    """ make file for submit """
    Xsubmit = load_sparse_csr(os.path.join(loaddir,'Xtest.npz'))
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


def main():

    X = load_sparse_csr(os.path.join(loaddir,'Xtrain.npz'))
    y = np.load(os.path.join(loaddir,'ytrain.npy'))
    # C = np.logspace(-4, 0, 10)
    C = [0.08]

    best_clf, best_score, best_C = None, float('Inf'), 0

    for c in C:
        clf, score = softmax_clf(c, X, y)
        if score < best_score:
            best_clf, best_score, best_C = clf, score, c

    print('\nBest score: {} at C = {}'.format(best_score, best_C))

    make_submit(clf)
    print('\nDone')
    

if __name__ == '__main__':
    main()

    
