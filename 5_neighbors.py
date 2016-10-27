import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import jaccard_similarity_score
import operator
from scipy import spatial

def neighbor(input_array,input_number):    # number = total neighbors input
    cust_i = input_array.shape[0]
    neighbor_output_name = np.zeros(shape=(cust_i,input_number+1), dtype='object')
    neighbor_output_score = np.zeros(shape=(cust_i,input_number+1), dtype='object')
    print ("Test from function ===== ")
    temp_array = np.zeros(shape=(cust_i,2), dtype='object')
    # print (temp_array)

    temp_array[:,0] = np.transpose(cust_ind[:])
    temp_array[:,1] = np.transpose(input_array[:,0])
    # temp_array[1] = np.transpose.input_array[1]
    # print (input_array[:,0])

    # cust_table_sorted = sorted(cust_table, key = operator.itemgetter(0,1))
    # cust_pre_table = np.array(cust_table_sorted, dtype='object')
    print(temp_array)

    neighbor_sorted = sorted(temp_array, key = operator.itemgetter(1), reverse=True)
    top_neighbor_array = np.array(neighbor_sorted, dtype='object')

    ### Get the data from ndarray so that the output will not be in the list
    ### you cannot use neighbor_output[0,0] = cust_ind[0] because you will get return for full object ex. array ['1000'],
    # not just a value of the list
    neighbor_output_name[0,0] = cust_ind.item(0)
    neighbor_output_name[0,1:] = top_neighbor_array[1:6,0]


    print (neighbor_output_name[0])

    neighbor_output_score[0,0] = cust_ind.item(0)
    neighbor_output_score[0,1:] = top_neighbor_array[1:6,1]

    print (neighbor_output_score[0])




df_input = pd.read_csv('/Users/suesalito/Desktop/recommedation/cosine_index_score_new.csv', header=0, low_memory=False, dtype=float)
df_invt = pd.read_csv('/Users/suesalito/Desktop/recommedation/inventoryid.csv', header=None, low_memory=False, dtype='object')
df_cust = pd.read_csv('/Users/suesalito/Desktop/recommedation/customerid.csv', header=None, low_memory=False, dtype='str')

index_array = df_input.iloc[0:,0:].values
print (index_array)

invt_ind = df_invt.iloc[0:,0:].values
print (invt_ind)

cust_ind = df_cust.iloc[0:,0:].values
print (cust_ind)

print()
print("# X nearest neighbors calculation ----------")


np.set_printoptions(threshold=np.inf)
print (index_array[:,0])
neighbor(index_array,5)



