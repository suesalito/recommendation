import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import pandas as pd
import operator

# testgit import

df = pd.read_csv('/Users/shying/Desktop/sales_data_sep_2016_delete00.csv', header=None, low_memory=False,
                 names=['Custid','Invtid','qty'],
                 dtype={'Custid': str, 'Invtid': int, 'qty':float})

# print(df)
# Define Arrays base on the data
y = df.iloc[0:10,:].values # get the first column row 0 to 9 in index ---> mean index 0 with 10 times
cust_table = df.iloc[0:,0:].values

# N=y.shape[0]
#
# for i in range(0,N):
#     try:
#         # print ('test')
#         # remove the space in the data with strip function r = right
#          y[i,0] = y[i,0].rstrip()
#
#     except ValueError:
#         print ('error on line :',i)
#
# print(y)



# Sort table by first and second key then transfer to the new array
s2 = sorted(y, key = operator.itemgetter(0,1))
order_array = np.array(s2, dtype='object')

N = cust_table.shape[0]

print(N)
print(cust_table)

# Check zero qty
c = 0

for i in range(0,N):
    try:
        # remove the space in the data with strip function r = right
         cust_table[i,0] = cust_table[i,0].rstrip()
         if (cust_table[i,2] == 0.0):
             print (cust_table[i])
             c += 1

    except ValueError:
        print ('error on line :',i)

print("c =",c) # if c = 0 means no zero ordqty

print(cust_table)

print ("====== After sort =======")
print ()
cust_table_sorted = sorted(cust_table, key = operator.itemgetter(0,1))
cust_pre_table = np.array(cust_table_sorted, dtype='object')
print(cust_pre_table)

# Get the unique invtid and custid from the list of transactions
invt_ind = np.unique(cust_pre_table[:,1])       # Get all material ids
print (invt_ind)
print (invt_ind.shape[0])

cust_ind = np.unique(cust_pre_table[:,0])       # Get all customer ids
print (cust_ind)
print (cust_ind.shape[0])

# ***** append a new row in to the same array

# newrow = ['100000TT',2,3]
# cust_pre_table = np.vstack([cust_pre_table, newrow])
# print ("Test add new row to the array"
#        "The result is ----")
# print (cust_pre_table)

# you code start here
# targets :
#
# create the 0,1 table base on the customer_ind and invt_ind by searching the data from cust_pre_table
#

# Example of creation array 2*2
# np.ndarray(shape=(2,2), dtype=float, order='F')

# Create the array base on customer number and total products

cust_i = cust_ind.shape[0]
invt_j = invt_ind.shape[0]
train_array = np.ndarray(shape=(cust_i,invt_j), dtype=float)
print(train_array)

c_index = 0         # for dataset scanning (cust)
i_index = 0         # for dataset scanning (invt)
c_current = ''      # current customer
i_cuurent = 0       # current invt
c_current = cust_pre_table[c_index,0]             # First customer
i_cuurent = cust_pre_table[i_index,1]             # First invt

print (invt_j)

# for k in range (0,100):
#     print (cust_pre_table[k])

for i in range(0,cust_i):
# for i in range(0,5):      ### **** use it incase you want to run just first 2 customers
    try:
        # remove the space in the data with strip function r = right
        #  cust_table[i,0] = cust_table[i,0].rstrip()
        #  if (cust_table[i,2] == 0.0):
        #      print (cust_table[i])
        #      c += 1
                        # get the current customer from the dataset
        # print()
        # print(cust_ind[i])
        for j in range(0,invt_j):

            # print("cust_pre_table[",c_index,",1] = ",cust_pre_table[c_index,1])
            # print("invt_ind]",j,"] = ",invt_ind[j])
            if(cust_pre_table[c_index,1] == invt_ind[j]) and (cust_pre_table[c_index,0] == cust_ind[i]):

                # To print out the matching date ====================

                # print("cust_pre_table[",c_index,",1] = ",cust_pre_table[c_index,1])
                # print("invt_ind[",j,"] = ",invt_ind[j])
                # if cust_pre_table[c_index,0] == '9999501':
                #     print (cust_pre_table[c_index,1])
                train_array[i,j] = 1
                if(c_index+1 < N):
                    c_index = c_index +1
                while True:
                    if((cust_pre_table[c_index,1] == cust_pre_table[c_index-1,1]) and (c_index+1 < N)
                        and (cust_pre_table[c_index,0] == cust_pre_table[c_index-1,0])):
                        # if cust_pre_table[c_index,0] == '9999501':
                        #     print (cust_pre_table[c_index,1])
                        c_index = c_index +1
                    else:
                        break

            # if(cust_pre_table[c_index+1,1] != invt_ind[j]):
            #     c_index += 1

    except ValueError:
        print ('error on line :',i)

# ******** To set the print out untruncated representation
np.set_printoptions(threshold=np.inf)

# Test print first customer and 2nd customer array 0 1
# print (train_array[0])
# print ("----------------------")
# print (train_array[1])


# print (train_array[1,37])
# print (invt_ind[37])
#
# print (cust_ind[1000])
# print (train_array[1000,93])
# print(invt_ind[93])

# print (cust_ind[1016])
# print (train_array[1016])

# for i in range(0,invt_j):
#     if(train_array[1016,i] == 1):
#         print(invt_ind[i])

# print (cust_ind[1017])

print()
print("# Test Jaccard")
# Test Jaccard
from sklearn.metrics import jaccard_similarity_score
y_pred = [0, 2, 1, 3,0,0,0,0,0,0,0,0]
y_true = [0, 1, 2, 3,0,0,0,0,0,0,0,0]
print (jaccard_similarity_score(y_true, y_pred))
y_pred33 = [0, 2, 1, 3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_true33 = [0, 1, 2, 3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print (jaccard_similarity_score(y_true33, y_pred33))
# x_pred = [train_array[i]]
# x_true = [train_array[j]]
# print (jaccard_similarity_score(x_true, x_pred))



# Calculate jacard similarity based on the train_array
# print (cust_ind[1015])
print (jaccard_similarity_score(train_array[1015], train_array[1016]))
# print(train_array[1015])
print("# Test Jaccard 2")
# Test Jaccard
cust_i = cust_ind.shape[0]
cust_j = cust_ind.shape[0]
jaccard_output = np.zeros(shape=(cust_i,cust_j), dtype=float)
# print(jaccard_output)

# for i in range(0,cust_i):
for i in range(0,5):
     try:
         for j in range(0,cust_j):
            # print (jaccard_similarity_score(np.array([train_array[1016]]), np.array([train_array[1016]])))
            jaccard_output[i,j] = jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]]))

     except ValueError:
         print ('error on line :', i)

        # print (jaccard_similarity_score(jaccard_output[i], jaccard_output[j]))
        # print (jaccard_similarity_score(jaccard_output[i], jaccard_output[j]))

# print (jaccard_output[0])

print ('--------------- 0 -> 19')
print cust_ind[100]
# for i in range (0,20):
for i in range(100, 1017):
    for j in range (0,cust_i):
        if ((jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]]))) > 0.50)\
                and ((jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]])))< 1.00):
            print cust_ind[i],cust_ind[j],(jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]])))
            # print (jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]])))

print ('--------------- 20 -> 30')

# for i in range (20, 10):
#     for j in range (0,cust_i):
#         if ((jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]]))) > 0.25):
#             print (cust_ind[i],' and',cust_ind[j])
#             print (jaccard_similarity_score(np.array([train_array[i]]), np.array([train_array[j]])))
#
#

# print (jaccard_output[123])
#
# for i in range (0,cust_i):
#     if jaccard_output[123,i] >= 0.981:
#         print (cust_ind[i],i)
#         print(train_array[i])
#         # print('=========')
# print (jaccard_similarity_score(train_array[1016], train_array[1016]))
#
# print (cust_ind)
# print (invt_ind)
# print (train_array[0])

# print (jaccard_similarity_score(train_array[123], train_array[567]))
# print (train_array[123])
# print()
# print(train_array[567])

# print ('-------------------------------------------------------------')
# print (jaccard_similarity_score(np.array([[0,0,0,0,1,1, 1,0, 0,1,0,0,0,1,1]]), np.array([[0,0,0,0,1,1, 0,0, 0,0,1,0,0,1,1]])))
#
# print ('-------------------------------------------------------------')
# print (jaccard_similarity_score(train_array[1015], train_array[1016]))
# print('====xxxxxx=====')
# print (jaccard_similarity_score(np.array([train_array[1015]]), np.array([train_array[1016]])))
