# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 00:08:56 2016

@author: Kun Wang
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np

peek = True

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
    

if peek: 
    print('\n==============================================')    
    print('Merging ', file5, ' and ', file6)
app_label_category = pd.merge(app_labels, label_categories, left_index='label_id', right_index='label_id',
                how='left', suffixes=['','_'])
       
app_label_category = app_label_category.drop('label_id_', axis=1)
app_label_category = app_label_category.dropna()
app_label_category.to_csv('../processed_data/merged_app_category.csv', encoding='utf-8')    
     
if peek:
    print(app_label_category.shape)