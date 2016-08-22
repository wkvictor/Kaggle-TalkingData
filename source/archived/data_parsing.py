# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:51:16 2016

@author: Victor
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np

peek = True

# ==============================================
file1 = '../input/gender_age_train.csv'
if peek: 
    print('\n==============================================')
    print('Loading ', file1)
gender_age_train = pd.read_csv(file1, encoding='utf-8', dtype={'device_id': np.str})
if peek: 
    print(gender_age_train.head())
    print('Shape:', gender_age_train.shape)      #  (74645, 4)
    print('Unique rows:', gender_age_train["device_id"].nunique())   # 74645
    
    
# ==============================================
file2 = '../input/phone_brand_device_model.csv'
if peek:
    print('\n==============================================')
    print('Loading', file2)
phone_brand_device_model = pd.read_csv(file2, encoding='utf-8', dtype={'device_id': np.str})
if peek: 
    print(phone_brand_device_model.head())
    print('Shape:', phone_brand_device_model.shape)        # (187245, 3)
    print('Unique rows:', phone_brand_device_model["device_id"].nunique())   # 186716
    
## Caution: 187245 rows, but unique device_id only 186716
phone_brand_device_model = phone_brand_device_model.drop_duplicates('device_id', keep='first')
if peek: print('Shape after removing dup:', phone_brand_device_model.shape)       # (186716, 3)
    

# ==============================================
if peek: 
    print('\n==============================================')    
    print('Merging ', file1, ' and ', file2)
merged_data = gender_age_train.merge(phone_brand_device_model, on='device_id', how='left')
if peek: 
    print(merged_data.head())
    print('Shape of merged data:', merged_data.shape)        # (74645, 6)
#    print(merged_data.describe())
    
del gender_age_train, phone_brand_device_model


# ==============================================
file3 = '../input/events.csv'
if peek:
    print('\n==============================================')
    print('Loading', file3)
events = pd.read_csv(file3, parse_dates=["timestamp"], encoding='utf-8',\
         dtype={'device_id': np.str, 'event_id': np.str})
if peek: 
    print(events.head())
    print('Shape:', events.shape)       # (3252950, 5)
    print('Unique device id:', events["device_id"].nunique())   # 60865
    print('Unique event id:', events["event_id"].nunique())    # 3252950
    
    
# ==============================================
if peek: 
    print('\n==============================================')    
    print('Merging merged_data and ', file3)
merged_data = merged_data.merge(events, on='device_id', how='inner')
if peek:
    print(merged_data.head())
    print('Shape of merged data:', merged_data.shape)   # (1215595, 10) Inflated
    
del events
    
      
# ==============================================
file4 = '../input/app_events.csv'
if peek: 
    print('\n==============================================')
    print('Loading', file4)
app_events = pd.read_csv(file4, encoding='utf-8', 
                         dtype={'event_id':np.str, 'app_id':np.str,
                         'is_active': np.int, 'is_installed': np.int})
if peek:
    print(app_events.head())
    print('Shape:', app_events.shape)
    print('Unique event id:', app_events['event_id'].nunique())
    print('Unique app id:', app_events['app_id'].nunique())
    
    
# ==============================================
if peek: 
    print('\n==============================================')    
    print('Merging merged_data and ', file4)
merged_data = merged_data.merge(app_events, on='event_id', how='inner')
if peek:
    print(merged_data.head())
    print('Shape of merged data:', merged_data.shape)   # (1215595, 10) Inflated
#    

   
# ==============================================
file5 = '../input/app_labels.csv'
if peek:
    print('\n==============================================')
    print('Loading', file5)
app_labels = pd.read_csv(file5, encoding='utf-8', dtype={'label_id': np.str, 'app_id': np.str})
if peek:
    print(app_labels.head())
    print('Shape:', app_labels.shape)
    print('Unique label id:', app_labels['label_id'].nunique())
    print('Unique app id:', app_labels['app_id'].nunique())


# ==============================================
file6 = '../input/label_categories.csv'
if peek:
    print('\n==============================================')
    print('Loading ', file6)
label_categories = pd.read_csv(file6, encoding='utf-8', dtype={'label_id': np.str, 'category': np.str})
if peek:
    print(label_categories.head())
    print('Shape:', label_categories.shape)
    print('Unique label id:', label_categories['label_id'].nunique())
    print('Unique category:', label_categories['category'].nunique())
#
## === merge 4 ===
#if peek: print('\n Merging ', file5, ' and ', file6)
#merged_data_temp = app_labels.merge(label_categories, on='label_id', how='left')
#if peek: 
#    print(merged_data_temp.head())
#    print(merged_data_temp.describe())
#
#del app_labels, label_categories 
#
#    
## === merge 5 ===
#if peek: print('\n Final Merging ...')
#merged_data.merge(merged_data_temp, on='app_id', how='left')
#if peek: 
#    print(merged_data.head())
#    print(merged_data.describe())
#    
#del merged_data_temp
#
#        
#merged_data.to_csv('../processed_data/merged_raw_data.csv', encoding='utf-8')
