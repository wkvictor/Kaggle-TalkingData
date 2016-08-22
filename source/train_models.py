from __future__ import print_function
import pandas as pd
import numpy as np
import os
#import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

loaddir = '../processed_data/'

# load sparse matrix
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix(( loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
    
Xsubmit = load_sparse_csr(os.path.join(loaddir,'Xtest.npz'))

X = load_sparse_csr(os.path.join(loaddir,'Xtrain.npz'))
y = np.load(os.path.join(loaddir,'ytrain.npy'))

# Train models
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=5183)

clf = LogisticRegression(C=0.01, multi_class='multinomial', solver='lbfgs')
clf.fit(Xtrain, ytrain)
ypred = clf.predict_proba(Xtest)
score = log_loss(ytest, ypred)   # -sum(y_true*log(y_pred_prob)) : the lower the better


# Submit format
clf.fit(X, y)
ysubmit = clf.predict_proba(Xsubmit)

row_index = np.load(os.path.join(loaddir,'submit_row_index.npy'))
col_name = np.load(os.path.join(loaddir,'submit_col_name.npy'))
pred = pd.DataFrame(ysubmit, index=row_index, columns=col_name)
pred.head()

pred.to_csv(os.path.join(loaddir,'submission.csv'),index=True)
