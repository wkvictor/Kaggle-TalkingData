# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 00:40:17 2016

@author: Kun Wang
"""

import pandas
import numpy as np
import os

os.chdir('../input/')

app_events=pandas.read_csv('app_events.csv', na_values=' ',dtype={'device_id':np.str, 'app_id':np.str}) 
app_labels=pandas.read_csv('app_labels.csv', na_values=' ',dtype={'label_id':np.str, 'app_id':np.str}) 
events=pandas.read_csv('events.csv', na_values=' ' , dtype={'device_id':np.str}) 
gender_age_train=pandas.read_csv('gender_age_train.csv', na_values=' ', dtype={'device_id':np.str}) 
gender_age_test=pandas.read_csv('gender_age_test.csv', na_values=' ', dtype={'device_id':np.str}) 
label_categories=pandas.read_csv('label_categories.csv', na_values=' ', dtype={'label_id':np.str}) 
phone_brand_device_model=pandas.read_csv('phone_brand_device_model.csv', na_values=' ',dtype={'device_id':np.str}) 


# merge events and app_labels on event id
data=pandas.merge(events,app_events,left_on='event_id',right_on='event_id', how='left')


# merge gender_age and phone data on device_id
data=pandas.merge(data,gender_age_train,left_on='device_id',right_on='device_id', how='left')


#data=pandas.merge(data,gender_age_test,left_on='device_id',right_on='device_id', how='left')


data=pandas.merge(data,phone_brand_device_model,left_on='device_id',right_on='device_id', how='left')


# merge app labels on app_id
data=pandas.merge(data,app_labels,left_on='app_id',right_on='app_id', how='left')


# merge label_categories on label_id
data=pandas.merge(data,label_categories,left_on='label_id',right_on='label_id', how='left')



data.count()