import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import jaccard_similarity_score
import operator



df_train = pd.read_csv('/Users/suesalito/Desktop/recommedation/output.csv', header=None, low_memory=False, dtype=int)

df_invt = pd.read_csv('/Users/suesalito/Desktop/recommedation/inventoryid.csv', header=0, low_memory=False, dtype='object')

df_cust = pd.read_csv('/Users/suesalito/Desktop/recommedation/customerid.csv', header=0, low_memory=False, dtype='object')

train_array = df_train.iloc[0:,0:].values
print (train_array)

invt_ind = df_invt.iloc[0:,0:].values
print (invt_ind)

cust_ind = df_cust.iloc[0:,0:].values
print (cust_ind)

print()
print("# Jaccard index calculation ----------")
# Jaccard code start

cust_i = cust_ind.shape[0]
cust_j = cust_ind.shape[0]
jaccard_output = np.zeros(shape=(cust_i,cust_j), dtype=float)

# for i in range(0,100):              # try running 100 customers
# ***  Uncomment if you wanna run all the jaccard again for all customers
for i in range(0,cust_i):
     try:
         for j in range(0+i,cust_j):
            # print (jaccard_similarity_score(np.array([train_array[1016]]), np.array([train_array[1016]])))
            jaccard_output[j,i] = jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]]))

     except ValueError:
         print ('error on line :', i)

# ***************************

print (jaccard_output)

import csv

# Create the csv file for the train_input

fl = open('/Users/suesalito/Desktop/recommedation/jaccard_index_score_new.csv', 'w')

writer = csv.writer(fl)

cust_ind_t = np.transpose(cust_ind)         # Transpose the customer array to concat as a header.
# writer.writerow(cust_ind_t)

jaccard_output = np.vstack([cust_ind_t,jaccard_output])         # concat the header

for values in jaccard_output:
    writer.writerow(values)

fl.close()
