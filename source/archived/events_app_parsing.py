# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:52:06 2016

@author: Victor
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os

os.chdir('../input/')

events = pd.read_csv('events.csv', na_values=' ' , 
                     dtype={'device_id':np.str})
        
app_events = pd.read_csv('app_events.csv', na_values=' ', 
                         dtype={'device_id':np.str, 'app_id':np.str})
                         
events_and_apps = pd.merge(events, app_events, how='right', on='event_id')
#events_and_apps.dropna(subset=['device_id'], inplace=True)
print(events_and_apps.count())