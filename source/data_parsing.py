# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:51:16 2016

@author: Victor
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np

peek = True

# === load 1 ===
file1 = '../input/gender_age_train.csv'
if peek: print('\n', file1)
gender_age_train = pd.read_csv(file1, encoding='utf-8', dtype={"device_id": np.str})
if peek: 
    print(gender_age_train.head())
    print(gender_age_train["device_id"].nunique())
    
file2 = '../input/events.csv'
if peek: print('\n', file2)
events = pd.read_csv(file2, encoding='utf-8', dtype={"event_id": np.str, "device_id": np.str}, 
                         parse_dates=["timestamp"])
if peek: 
    print(events.head())
    print(events["device_id"].nunique())

# === merge 1 ===
if peek: print('\n === Merge 1 ===')
cleaned_data = gender_age_train.merge(events, on='device_id', how='inner')
if peek: 
    print(cleaned_data.head())
    print(cleaned_data["device_id"].nunique())

del gender_age_train, events
    

# === load 2 ===
file3 = '../input/phone_brand_device_model.csv'
if peek: print('\n', file3)
phone_brand_device_model = pd.read_csv(file3, encoding='utf-8', dtype={"device_id": np.str})
if peek: 
    print(phone_brand_device_model.head())
    print(phone_brand_device_model["device_id"].nunique())
    
# === merge 2 ===
if peek: print('\n === merge 2 ===')
cleaned_data = cleaned_data.merge(phone_brand_device_model, on='device_id', how='left')
if peek: 
    print(cleaned_data.head())
    print(cleaned_data["device_id"].nunique())
    
##### To-do: map phone brand and model to numeric numbers #####
    
#file4 = '../input/app_events.csv'
#app_events = pd.read_csv(file4, encoding='utf-8', dtype={"event_id": np.str, "app_id": np.str})

        

 

