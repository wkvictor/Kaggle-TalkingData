from __future__ import print_function
import pandas as pd
import numpy as np
import os

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder


datadir = '../input'

# load data
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col='device_id')

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
phone = phone.drop_duplicates('device_id', keep='first').set_index('device_id')

# Add index column for creating sparse matrix: one-hot encoder
gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# Encode phone brand
brand_encoder = LabelEncoder()
brand_encoder.fit(phone['phone_brand'])
phone['brand'] = brand_encoder.transform(phone['phone_brand'])   # numeric labels

# phone is larger: join two structures use '=' only
# extract part of dataframe according to index 'device_id'
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']

# sparse matrix: csr_matrix((data,(row,col)))
Xtrain_brand = csr_matrix((np.ones(gatrain.shape[0]), (gatrain['trainrow'], gatrain['brand'])))
Xtest_brand = csr_matrix((np.ones(gatest.shape[0]), (gatest['testrow'], gatest['brand'])))
print('Brand matrix: Train shape {}; Test shape {}'.format(Xtrain_brand.shape, Xtest_brand.shape))

# Encode device model
model_encoder = LabelEncoder()
model_encoder.fit(phone['device_model'])
phone['model'] = model_encoder.transform(phone['device_model'])

gatrain['model'] = phone['model']
gatest['model'] = phone['model']

Xtrain_model = csr_matrix( (np.ones(gatrain.shape[0]), (gatrain['trainrow'], gatrain['model'])))
Xtest_model = csr_matrix( (np.ones(gatest.shape[0]), (gatest['testrow'], gatest['model'])))
print('Model matrix: Train shape {}; Test shape {}'.format(Xtrain_model.shape, Xtest_model.shape))


events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')
app_events = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})    # natural index: 0~len-1
app_labels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

# For each device, mark which apps it has installed. 
# So there are as many feature columns as there are distinct apps.
app_encoder = LabelEncoder().fit(app_events.app_id)
app_events['app'] = app_encoder.transform(app_events.app_id)
napps = len(app_encoder.classes_)

device_apps = (app_events.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
#print(device_apps.head())                      

d = device_apps.dropna(subset=['trainrow'])
Xtrain_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), shape=(gatrain.shape[0], napps))

d = device_apps.dropna(subset=['testrow'])
Xtest_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), shape=(gatest.shape[0], napps))

print('App matrix: Train shape {}; Test shape {}'.format(Xtrain_app.shape, Xtest_app.shape))

# extract rows that are in app_events
app_labels = app_labels.loc[app_labels.app_id.isin(app_events.app_id.unique())]
app_labels['app'] = app_encoder.transform(app_labels.app_id)
label_encoder = LabelEncoder().fit(app_labels.label_id)
app_labels['label'] = label_encoder.transform(app_labels.label_id)
nlabels = len(label_encoder.classes_)

device_labels = (device_apps[['device_id','app']]
                .merge(app_labels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
                
d = device_labels.dropna(subset=['trainrow'])
Xtrain_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0], nlabels))
d = device_labels.dropna(subset=['testrow'])
Xtest_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtrain_label.shape, Xtest_label.shape))


# concatenate all feature matrices
Xtrain = hstack((Xtrain_brand, Xtrain_model, Xtrain_app, Xtrain_label), format='csr')
Xtest =  hstack((Xtest_brand, Xtest_model, Xtest_app, Xtest_label), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


target_encoder = LabelEncoder().fit(gatrain.group)
y = target_encoder.transform(gatrain.group)


# save sparse matrix: each dim is a numpy array
def save_sparse_csr(filepath, arr):
    np.savez(filepath, data=arr.data, indices=arr.indices, indptr=arr.indptr, shape=arr.shape)
    
savedir = '../processed_data'
save_sparse_csr(os.path.join(savedir,'Xtrain.npz'), Xtrain)
save_sparse_csr(os.path.join(savedir,'Xtest.npz'), Xtest)

np.save(os.path.join(savedir,'ytrain.npy'), y)

# save submit info
submit_index = gatest.index.values   # np array
np.save(os.path.join(savedir,'submit_row_index.npy'), submit_index)
np.save(os.path.join(savedir,'submit_col_name.npy'), target_encoder.classes_)




