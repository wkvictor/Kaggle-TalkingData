# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:51:16 2016

@author: Victor
"""

from __future__ import print_function, division
import pandas as pd


def test_load(filepath):
	
	df = pd.read_csv(filepath)
	print(df.head())
	print(df.describe())
		
	
def main_load():
	
	file1 = '../input/gender_age_train.csv'
	print('\n', file1)
	test_load(file1)
					
	file2 = '../input/gender_age_test.csv'
	print('\n', file2)
	test_load(file2)
	
	file3 = '../input/phone_brand_device_model.csv'
	print('\n', file3)
	test_load(file3)
	
	file4 = '../input/events.csv'
	print('\n', file4)
	test_load(file4)
	
	
	
def test_join(file1, file2):
	
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
	
    df = df1.merge(df2, on='device_id', how='left')
    df.to_csv('../processed_data/gender_age_phone_brand_device_model.csv')
	
    print(df.head())
 

def main_join():
	
	file1 = '../input/gender_age_train.csv'
	file2 = '../input/phone_brand_device_model.csv'
#	file3 = './input/events.csv'
	
	test_join(file1, file2)	
 
 
 
#	test_join(file1, file3)
	
	
	
if __name__ == '__main__':
#	main_load()
    main_join()