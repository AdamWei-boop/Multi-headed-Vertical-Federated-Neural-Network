# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:43:55 2021

@author: adamwei
"""

import tensorflow as tf
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from tensorflow.keras import optimizers, losses
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
# https://imbalanced-learn.org/stable/
from avazudataset import AvazuDataset
from utils import create_criteo_dataset, load_dat, batch_split
from tf_model import tf_organization_graph, tf_top_graph, tf_graph
from quantization_sche import quant_process

def main(args):
    
    data_type = args.data_type           # Define the data options: 'original', 'encoded'
    model_type = args.model_type         # Define the learning methods: 'vertical', 'centralized'
    epochs = args.epochs                 # number of training epochs
    nrows = 50000                        # subselection of rows to speed up the program
    
    # Ming added the following variables for configurable vertical FL
    organization_num = args.organization_num    # number of participants in vertical FL
    attribute_split_array = \
        np.zeros(organization_num).astype(int)  # initialize a dummy split scheme of attributes
    
    # dataset preprocessing
    if data_type == 'original': 
        
        if args.dname == 'adult':
            
            file_path = "./datasets/{0}.csv".format(args.dname)
            X = pd.read_csv(file_path)
            # X = pd.read_csv(file_path, nrows = nrows)
        
            # Ming added the following codes on 11/11/2021 to pre-process the Adult dataset
            # remove a duplicated attributed of "edu"
            X = X.drop(columns=['edu'])
            # rename an attribute from "skin" to "race"
            X = X.rename({'skin':'race'}, axis=1)
            
            # Ming added the following codes on 11/11/2021 to perform a quick sanity check of the dataset
            X.head()
            for attribute in X.columns:
                #print(dt.value_counts(dt[attribute]))
                plt.figure()
                X.value_counts(X[attribute]).sort_index(ascending=True).plot(kind='bar')
  
            N, dim = X.shape
          
            # Print out the dataset settings
            print("\n\n=================================")
            print("\nDataset:", args.dname, \
                  "\nNumber of attributes:", dim-1, \
                  "\nNumber of labels:", 1, \
                  "\nNumber of rows:", N)        
          
            columns = list(X.columns)
            
            # get the attribute data and label data
            y = X['income'].values.astype('int')
            X = X.drop(['income'], axis=1)
            
            # set up the attribute split scheme for vertical FL
            attribute_split_array = \
                np.ones(len(attribute_split_array)).astype(int) * \
                int((dim-1)/organization_num)
                
            # correct the attribute split scheme if the total attribute number is larger than the actual attribute number
            if np.sum(attribute_split_array) > dim-1:
                print('unknown error in attribute splitting!')
            elif np.sum(attribute_split_array) < dim-1:
                missing_attribute_num = (dim-1) - np.sum(attribute_split_array)
                attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
            else:
                print('Successful attribute split for multiple organizations')
                
                
        elif args.dname == 'avazu':
            
            file_path = './datasets/{0}.gz'.format(args.dname)
            df = pd.read_csv(file_path, compression='gzip', nrows=nrows)
            df.to_csv('./datasets/{0}.csv'.format(args.dname), index=False)
            
            columns = df.columns.drop('id')
            # data = pd.read_csv(file_path, compression='gzip', nrows=nrows)
            data = AvazuDataset('./datasets/{0}.csv'.format(args.dname), rebuild_cache=True)
            
            # X = data.fillna('-1')
            
            # X.head()

            data, labels = [data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))]
            labels = np.reshape(labels, [len(labels), 1])
            data = np.concatenate((labels, data),axis=1)
            
            X = pd.DataFrame(data, columns=columns)
            
            # get the attribute data and label data
            y = X['click'].values.astype('int')
            X = X.drop(['click'], axis=1)
            
            N, dim = X.shape
               
             # Print out the dataset settings
            print("\n\n=================================")
            print("\nDataset:", args.dname, \
                   "\nNumber of attributes:", dim, \
                   "\nNumber of labels:", 1, \
                   "\nNumber of rows:", N,\
                   "\nPostive ratio:", sum(y)/len(y))            
            
            columns = list(X.columns)
            
             # set up the attribute split scheme for vertical FL
            attribute_split_array = \
                 np.ones(len(attribute_split_array)).astype(int) * \
                 int(dim/organization_num)
                 
             # correct the attribute split scheme if the total attribute number is larger than the actual attribute number
            if np.sum(attribute_split_array) > dim:
                print('unknown error in attribute splitting!')
            elif np.sum(attribute_split_array) < dim:
                missing_attribute_num = dim - np.sum(attribute_split_array)
                attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
            else:
                print('Successful attribute split for multiple organizations')
                                  
    else:
        file_path = "./dataset/{0}.dat".format(args.dname)
        X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)    
        
    ### A LR baseline
    
    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)

    # from sklearn.linear_model import LogisticRegression

    # logreg = LogisticRegression()

    # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    # grid = GridSearchCV(logreg, param_grid, cv=6).fit(X, y)

    # print("Logistic Regression: ", grid.best_score_, grid.best_params_)    
    
    # start the learning process
    print('\nLearning methods:', model_type)
    
    # initialize the arrays to be return to the main function
    loss_array = []
    acc_array = []
    test_epoch_array = []
    
    if model_type == 'vertical':
                
        # print out the vertical FL setting 
        print('\nThe current vertical FL has a non-configurable structure.')
        print('Reconfigurable vertical FL can be achieved by simply changing the attribute group split!')
        
        # Ming revised the following codes on 12/11/2021 to realize re-configurable vertical FL
        print('Ming revised the codes on 12/11/2021 to realize re-configurable vertical FL.')       
        print('\nThere are {} participant organizations:'.format(organization_num))
        
        # define the attributes in each group for each organization
        attribute_groups = []
        attribute_start_idx = 0
        for organization_idx in range(organization_num):
            attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
            attribute_groups.append(columns[attribute_start_idx : attribute_end_idx])
            attribute_start_idx = attribute_end_idx
            print('The attributes held by Organization {0}: {1}'.format(organization_idx, attribute_groups[organization_idx]))                        
        
        # get the vertically split data with one-hot encoding for multiple organizations
        vertical_splitted_data = {}
        encoded_vertical_splitted_data = {}
        chy_one_hot_enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        for organization_idx in range(organization_num):
            vertical_splitted_data[organization_idx] = \
                X[attribute_groups[organization_idx]].values.astype('float32')
            encoded_vertical_splitted_data[organization_idx] = \
                chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
            print('The shape of the encoded dataset held by Organization {0}: {1}'.format(organization_idx, np.shape(encoded_vertical_splitted_data[organization_idx])))                       
        
        # set up the random seed for dataset split
        random_seed = 1001
        
        # split the encoded data samples into training and test datasets
        X_train_vertical_FL = {}
        X_test_vertical_FL = {}
        
        for organization_idx in range(organization_num):
            if organization_idx == 0:
                X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], y_train, y_test = \
                    train_test_split(encoded_vertical_splitted_data[organization_idx], y, test_size=0.2, random_state=random_seed)
            else:
                X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], _, _ = \
                    train_test_split(encoded_vertical_splitted_data[organization_idx], y, test_size=0.2, random_state=random_seed)

                    
        y_train = to_categorical(y_train, 2)   
        
        # set the neural network structure parameters
        organization_hidden_units_array = [np.array([32])]*organization_num

        organization_output_dim = np.array([64])
    
        top_hidden_units = np.array([64])
        top_output_dim = np.array([2])
        
        activation = 'relu'
        
        # build the client models
        organization_models = {}
        for organization_idx in range(organization_num):
            organization_models[organization_idx] = \
                tf_organization_graph(organization_hidden_units_array[organization_idx], \
                                organization_output_dim, activation)
             
        # build the top model over the client models
        top_model = tf_top_graph(top_hidden_units, top_output_dim, activation)
    
        # define the neural network optimizer
        optimizer = optimizers.Adam(learning_rate=0.002, name='Adam')

        #optimizer = optimizers.SGD(learning_rate=0.002, momentum=0)
    
        # conduct vertical FL
        print('\nStart vertical FL......\n')
        
        for i in range(epochs):
            
            batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), args.batch_size, args.batch_type)
            
            for batch_idxs in batch_idxs_list:
            
                with tf.GradientTape(persistent = True) as tape:            
        
                    organization_outputs = {}
                    for organization_idx in range(organization_num):
                        
                        forward_output = organization_models[organization_idx](X_train_vertical_FL[organization_idx][batch_idxs])

                        local_quant = quant_process(args.quant_sche, forward_output, args.quant_level, args.base_bits)
                        forward_output_incre, communication_cost, mse_error = local_quant.quant()
                        forward_output = forward_output + forward_output_incre
                        organization_outputs[organization_idx] = forward_output
                        
                    y_pre = top_model(organization_outputs)
                    
                    # original code: y_pre = top_model(client_output0, client_output1, client_output2)
                    # revised code version 1: y_pre = top_model(client_outputs[0], client_outputs[1], client_outputs[2])
                    # revised code version 2: y_pre = top_model([client_outputs[j] for j in range(len(client_outputs))])
    
                    loss = tf.reduce_mean(losses.categorical_crossentropy(y_true=y_train[batch_idxs], y_pred=y_pre))
    
                    top_grad = tape.gradient(loss, top_model.variables)
                    
                    optimizer.apply_gradients(grads_and_vars=zip(top_grad, top_model.variables))
            
                    # Kang: the following codes need additional work
                    for organization_idx in range(organization_num):
                        organization_model_grad = tape.gradient(loss, organization_models[organization_idx].variables)     
                        optimizer.apply_gradients(grads_and_vars=zip(organization_model_grad, organization_models[organization_idx].variables))
                    
                    del tape
                
            # let the program report the simulation progress
            if (i+1)%1 == 0:
                organization_outputs_for_test = {}
                for organization_idx in range(organization_num):
                    organization_outputs_for_test[organization_idx] = \
                        organization_models[organization_idx](X_test_vertical_FL[organization_idx])
                
                log_probs = top_model(organization_outputs_for_test)
                # pre = tf.argmax(log_probs, axis=1)
                # acc = accuracy_score(y_test, pre)
                acc = roc_auc_score(y_test, log_probs[:,1])
                print('For the {0}-th epoch, train loss: {1}, test acc: {2}'.format(i+1, loss.numpy(), acc))
                
                test_epoch_array.append(i+1)
                loss_array.append(loss.numpy())
                acc_array.append(acc)
    
    elif model_type == 'centralized':
        
        # one-hot encoding the original data
        chy_one_hot_enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        X = chy_one_hot_enc.fit_transform( X )
    
        print('Client data shape: {}, postive ratio: {}'.format(X.shape, sum(y)/len(y)))
        
        # split the data into training and test datasets
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=0)    

        # y_train = to_categorical(y_train, 2)

        hidden_units = [128, 32]
        output_dim = 1
        
        activation = 'relu'
        
        model = tf_graph(hidden_units, output_dim, activation)
    
        optimizer = optimizers.Adam(learning_rate=0.0002, name='Adam')
        
        # optimizer = optimizers.SGD(learning_rate=0.02, momentum=0)
    
        # conduct centrailized learning
        print('\nStart centralized learning......\n')
        for i in range(epochs):
            
            batch_idxs_list = batch_split(len(X_train), args.batch_size, args.batch_type)
            
            for batch_idxs in batch_idxs_list:
            
                with tf.GradientTape() as tape:
    
                    # SGD optimization of the neural network
                    logits = model(X_train[batch_idxs])
                    # print(y_pre.shape, y_train.shape)
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, 
                                        labels=tf.dtypes.cast(tf.reshape(y_train[batch_idxs], shape=[-1, 1]), tf.float32)))
                    # loss = tf.reduce_mean(losses.categorical_crossentropy(y_true=y_train[batch_idxs], y_pred=y_pre))
    
                    model_grad = tape.gradient(loss, model.variables)
                    optimizer.apply_gradients(grads_and_vars=zip(model_grad, model.variables))
        
            # let the program report the simulation progress
            if (i+1)%1 == 0:
                logits = tf.math.sigmoid(model(X_test))
                # pre = tf.argmax(log_probs, axis=1)
                # acc = accuracy_score(y_test, pre)
                auc = roc_auc_score(y_test, logits)
                print('For the {0}-th epoch, train loss: {1}, test auc: {2}'.format(i+1, loss.numpy(), auc))
                
                test_epoch_array.append(i+1)
                loss_array.append(loss.numpy())
                acc_array.append(auc)
    
    return test_epoch_array, loss_array, acc_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vertical FL')
    parser.add_argument('--dname', default='adult', help='dataset name: avazu, adult')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs') 
    parser.add_argument('--batch_type', type=str, default='mini-batch')  
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
    parser.add_argument('--model_type', default='vertical', help='define the learning methods: vrtical or centralized')    
    parser.add_argument('--organization_num', type=int, default=2, help='number of origanizations, if we use vertical FL')    
    parser.add_argument('--quant_sche', default='bucket_uniform')    
    parser.add_argument('--quant_level', type=int, default=2)
    parser.add_argument('--base_bits', type=int, default=16)      


    args = parser.parse_args()

    start_time = time.time()
    
    # collect experimental results
    test_epoch_array, loss_array, acc_array = main(args)

    elapsed = time.time() - start_time
    mins, sec = divmod(elapsed, 60)
    hrs, mins = divmod(mins, 60)
    print("Elasepd time: %d:%02d:%02d" % (hrs, mins, sec))
    
    # quickly plot figures to see the loss and acc performance
    figure_loss = plt.figure(figsize = (8, 6))
    plt.plot(test_epoch_array, loss_array)
    plt.xlabel('Communication round #')
    plt.ylabel('Test loss')
    plt.ylim([0, 1])
    
    
    figure_acc = plt.figure(figsize = (8, 6))
    plt.plot(test_epoch_array, acc_array)
    plt.xlabel('Communication round #')
    plt.ylabel('Test accuracy')
    plt.ylim([0, 1])    
    