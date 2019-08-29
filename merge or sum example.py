# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:27:30 2019

@author: suesa
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import pandas as pd
import operator


#download CSV to Panda
df = pd.read_csv(r'C:\Users\suesa\Desktop\Julyonhand\Eastlandonhand072619_2.csv', header=None, low_memory=False,
                 names=['Invtid','qty','AvgCost','TotalCost'],
                 dtype={'Invtid': str, 'qty':float,'AvgCost':float,'TotalCost':float})

print(df)
# Define Arrays base on the data
#y = df.iloc[0:10,:].values # get the first column row 0 to 9 in index ---> mean index 0 with 10 times
#cust_table = df.iloc[0:,0:].values


s2 = sorted(y, key = operator.itemgetter(0))
order_array = np.array(s2, dtype='object')

#print(order_array)
print(order_array.shape[0])

print('TEST')
print(order_array[0:10])


invt_ind = np.unique(order_array[:,0])       # Get all material ids
print (invt_ind)
#print (invt_ind.shape[0])

invt_j = invt_ind.shape[0]
#train_array = np.ndarray(shape=(invt_j,3), dtype=float)
train_array = np.zeros(shape=(invt_j,3), dtype=float)
#np.zeros(train_array)

#print(train_array)

#print (train_array.shape[0])
#print (train_array.shape[1])
print(invt_j)
print('-------------------')

#for i in range(0,cust_i):
invt_idx = -1
#for i in range(0,15):      ### **** use it incase you want to run just first 2 customers
for i in range(0,order_array.shape[0]):
    try:
         
         if(order_array[i,0] == order_array[i-1,0]) and i>0:
             #print ('Yes')
             #print('Yes',invt_ind[i],train_array[invt_idx])
             #print('Add',order_array[i,1:4])
             train_array[invt_idx,0] = train_array[invt_idx,0] + order_array[i,1]
             train_array[invt_idx,2] = train_array[invt_idx,2] + order_array[i,3]
             #print(order_array[i,1:4])
             #print(train_array[invt_idx])
         else:
             #print ('New record')
             invt_idx = invt_idx +1
             #print('SET',invt_ind[invt_idx],train_array[invt_idx],order_array[i,1:4])
             #print('test',order_array[i,1:4])
             train_array[invt_idx] = order_array[i,1:4]
             # Set avg cost column 0 because no use
             train_array[invt_idx,1] = 0
             #print('After set',invt_ind[invt_idx],train_array[invt_idx])
             #print('----end add new record')


    except ValueError:
        print ('error on line :',i)
        
        
        
print('-------------------')
print(train_array[0:invt_ind.shape[0]])
print(train_array.shape)



invt_ind_t = np.transpose(invt_ind)         # Transpose the customer array to concat as a header.

print (invt_ind)
np.transpose(invt_ind)

country = np.array(['USA', 'Japan', 'UK', '', 'India', 'China'], dtype = 'object') 
  
# Print the array 
print(country)


# =============================================================================
# foutput = np.hstack((invt_ind_t,train_array))   
# =============================================================================

output_array = np.zeros(shape=(invt_j,4), dtype=object)     
for i in range(0,output_array.shape[0]):
    output_array[i,0] = invt_ind[i]
    output_array[i,1:4] = train_array[i]


print (output_array)


import csv

# Create the csv file for the train_input

fl = open('/Users/suesa/Desktop/Julyonhand/outputnew.csv', 'w')

writer = csv.writer(fl)

for values in output_array:
    writer.writerow(values)
# # # 
fl.close()
print('Finish')
# =============================================================================
# =============================================================================
# =============================================================================






