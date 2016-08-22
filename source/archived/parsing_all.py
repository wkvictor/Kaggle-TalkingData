# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:15:30 2016

@author: Kun Wang
"""

import pandas as pd

table1to4 = pd.merge(ga_phone, events_and_apps, on='device_id', how='right')
table1to4.dropna(inplace=True)
table1to4.to_csv('../processed_data/merged_table_1_to_4.csv', encoding='utf-8')