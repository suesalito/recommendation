import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import jaccard_similarity_score
import operator
from scipy import spatial

def export_csv(rec_array_csv, path):
    import csv

    # Create the csv file for the train_input

    fl = open(path, 'w')

    writer = csv.writer(fl)

    # cust_ind_t = np.transpose(cust_ind)         # Transpose the customer array to concat as a header.
    # writer.writerow(cust_ind_t)


    for values in rec_array_csv:
        writer.writerow(values)

    fl.close()
    print ("File has been saved at :",path)

def neighbor(input_array,input_number,rec_number):    # number = total neighbors input
    cust_i = input_array.shape[0]
    print (cust_i)
    invt_j = invt_ind.shape[0]
    input_range = input_number +1
    rec_number_range = rec_number +1
    neighbor_output_name = np.zeros(shape=(cust_i,input_range), dtype='object')
    neighbor_output_score = np.zeros(shape=(cust_i,input_range), dtype='object')
    customer_rec_item = np.zeros(shape=(cust_i,rec_number_range), dtype='object')
    customer_rec_item_weight = np.zeros(shape=(cust_i,rec_number_range), dtype='object')
    customer_rec_item_score = np.zeros(shape=(cust_i,invt_j), dtype=int)
    print ("Test from function ===== ")
    temp_array = np.zeros(shape=(cust_i,2), dtype='object')
    temp_array_rec = np.zeros(shape=(invt_j,2), dtype='object')
    # print (temp_array)

    temp_array[:,0] = np.transpose(cust_ind[:])
    temp_array_rec[:,0] = np.transpose(invt_ind[:])

    for i in range (0,cust_ind.shape[0]):
    # for i in range (0,10):
        # print ("TEST ",input_array[:,i])
        temp_array[:,1] = np.transpose(input_array[:,i])
        # temp_array[1] = np.transpose.input_array[1]
        # print (input_array[:,0])

        # cust_table_sorted = sorted(cust_table, key = operator.itemgetter(0,1))
        # cust_pre_table = np.array(cust_table_sorted, dtype='object')
        # print(temp_array)

        neighbor_sorted = sorted(temp_array, key = operator.itemgetter(1), reverse=True)
        top_neighbor_array = np.array(neighbor_sorted, dtype='object')


        ### Get the data from ndarray so that the output will not be in the list
        ### you cannot use neighbor_output[0,0] = cust_ind[0] because you will get return for full object ex. array ['1000'],
        # not just a value of the list
        print ()
        print ("*** Start process", cust_ind[i])
        neighbor_output_name[i,0] = cust_ind.item(i)
        neighbor_output_name[i,1:] = top_neighbor_array[1:input_range,0]
        print (neighbor_output_name[i])

        neighbor_output_score[i,0] = cust_ind.item(i)
        neighbor_output_score[i,1:] = top_neighbor_array[1:input_range,1]
        print (neighbor_output_score[i])

        print ("*** Get the index of neighbors for customer", cust_ind[i])
        # itemindex = np.where(cust_ind==cust_ind.item(i))
        # print (itemindex[0][0])

        neighbor_index = np.zeros(shape=(cust_i,input_range), dtype=int)
        for j in range (0,input_range):
            itemindex = np.where(cust_ind==neighbor_output_name[i,j])
            # print (itemindex[0][0])
            neighbor_index[i,j] = itemindex[0][0]


        # base_cust_buy_index = np.nonzero(buy_index[neighbor_index[i,0]])
        base_cust_buy_index = np.nonzero(buy_index[neighbor_index[i]])
        ## base_cust_buy_index[1] returns the number of invtid were bought by all customers
        # print (base_cust_buy_index)        # See list of product
        cust_buy_dup_index = np.nonzero(buy_index[neighbor_index[i,0]])
        # print (cust_buy_dup_index)
        # base_cust_buy_index = np.nonzero(buy_index[neighbor_index[i]])
        # ******* (Count the quantity weight item from neighbors
        unique, counts = np.unique(base_cust_buy_index[1], return_counts=True)
        # print (dict(zip(unique, counts)))
        # print (unique)
        # print (counts)
        # This for loop return the same from nonzero function
        # for k in range(0,invt_ind.shape[0]-1):
        #     if buy_index[i,k] == 1:
        #         print (k)

        customer_rec_item_score[i,unique] = counts
        customer_rec_item_score[i,cust_buy_dup_index] = -1


        temp_array_rec[:,1] = np.transpose(customer_rec_item_score[i,:])
        # print (temp_array_rec)
        rec_item_sorted = sorted(temp_array_rec, key = operator.itemgetter(1), reverse=True)
        top_rec_array = np.array(rec_item_sorted, dtype='object')


        customer_rec_item[i,0] = cust_ind.item(i)
        customer_rec_item[i,1:] = top_rec_array[1:rec_number_range,0]
        customer_rec_item_weight[i,0] = cust_ind.item(i)
        customer_rec_item_weight[i,1:] = top_rec_array[1:rec_number_range,1]

        print (customer_rec_item[i])
        print (customer_rec_item_weight[i])
        # print (customer_rec_item_score[i])
        #

        # itemindex = np.where(customer_rec_item_score[i]==3)       # Sample code to print item match 3 times
        # print (itemindex[0])



        # print (neighbor_index[i])
        # print (buy_index[neighbor_index[i]])

        # print (neighbor_output_name[i])

        neighbor_output_score[i,0] = cust_ind.item(i)
        neighbor_output_score[i,1:] = top_neighbor_array[1:input_range,1]

        # print (neighbor_output_score[i])

    # print (customer_rec_item_score.shape[0])
    # print (customer_rec_item_score.shape[1])

    customer_rec_item_score = np.hstack((cust_ind,customer_rec_item_score))
    # print (invt_ind)
    invt_ind_header = np.vstack((['Table'],invt_ind))         # concat the header
    # print (invt_ind_header.shape[0])
    invt_ind_header_temp = np.transpose(invt_ind_header)
    # print (invt_ind_header.shape[0])

    customer_rec_item_score = np.vstack([invt_ind_header_temp,customer_rec_item_score])         # concat the header

    # print (customer_rec_item_score.shape[0])
    # print (customer_rec_item_score.shape[1])

    # neighbor_output_name
    export_csv(neighbor_output_name,'/Users/suesalito/Desktop/recommedation/cosine_neighbor_output_name4.csv')
    # neighbor_output_score
    export_csv(neighbor_output_score,'/Users/suesalito/Desktop/recommedation/cosine_neighbor_output_score4.csv')

    # Reccomendation_name
    export_csv(customer_rec_item,'/Users/suesalito/Desktop/recommedation/cosine_neighbor_rec_item4.csv')
    # Recommendation_qty_weight
    export_csv(customer_rec_item_weight,'/Users/suesalito/Desktop/recommedation/cosine_neighbor_rec_weight4.csv')

    # Recommendation system
    export_csv(customer_rec_item_score,'/Users/suesalito/Desktop/recommedation/cosine_rec_item_output4.csv')


df_buy_index = pd.read_csv('/Users/suesalito/Desktop/recommedation/output3.csv', header=None, low_memory=False, dtype=int)
df_invt = pd.read_csv('/Users/suesalito/Desktop/recommedation/inventoryid3.csv', header=0, low_memory=False, dtype='object')
df_cust = pd.read_csv('/Users/suesalito/Desktop/recommedation/customerid3.csv', header=0, low_memory=False, dtype='str')

# Select the Cosine or Jaccard index input.
df_input = pd.read_csv('/Users/suesalito/Desktop/recommedation/cosine_index_score_new_3.csv', header=0, low_memory=False, dtype=float)


index_array = df_input.iloc[0:,0:].values
# print (index_array)

for ii in range (index_array.shape[0]):
    for jj in range (index_array.shape[0]):
        if index_array[ii,jj] == 0:
            index_array[ii,jj] = index_array[jj,ii]
        if ii == jj:
            index_array[ii,jj] = 1.1

print (index_array)


invt_ind = df_invt.iloc[0:,0:].values
print (invt_ind)

cust_ind = df_cust.iloc[0:,0:].values
print (cust_ind)

buy_index = df_buy_index.iloc[0:,0:].values

print()
print("# X nearest neighbors calculation ----------")


np.set_printoptions(threshold=np.inf)
# print (index_array[:,0])
# print (index_array[-1])
neighbor(index_array,5,10) ## send number of neighbors and number or recommendation return



