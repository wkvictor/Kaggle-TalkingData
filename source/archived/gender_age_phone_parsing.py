# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 13:08:44 2016

@author: Kun Wang
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np
import os

os.chdir('../input/')

gender_age_train = pd.read_csv('gender_age_train.csv', encoding='utf-8',
                               dtype={'device_id':np.str, 'group':np.str})

phone_device = pd.read_csv('phone_brand_device_model.csv', encoding='utf-8',
                           dtype={'device_id':np.str, 'phone_brand':np.str,
                                  'device_model':np.str})
                                  
phone_device = phone_device.drop_duplicates('device_id', keep='first')
              
ga_phone = pd.merge(gender_age_train, phone_device, on='device_id', how='left')

print(ga_phone.count())