'''
# Time   : 2021/10/25 14:40
# Author : adamwei
# Vertical FL
'''
from deepFM_model import tf_host_bottom_graph,\
      tf_guest_bottom_graph, tf_top_graph
from utils import create_criteo_dataset_dfm
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    k = 10
    w_reg = 1e-4
    v_reg = 1e-4

    interval_dense, interval_sparse = 6, 13

    file_path = './dataset/criteo.txt'
        
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset_dfm(file_path, data_size=50000, test_size=0.2)
    
    X_train_host = np.hstack((X_train[:,:interval_dense], X_train[:,13:26]))
    X_train_guest = np.hstack((X_train[:,interval_dense:13], X_train[:,26:]))

    X_test_host = np.hstack((X_test[:,:interval_dense], X_test[:,13:26]))
    X_test_guest = np.hstack((X_test[:,interval_dense:13], X_test[:,26:]))

    feat_host = [feature_columns[0][:interval_dense],feature_columns[1][:interval_sparse]]
    feat_guest = [feature_columns[0][interval_dense:],feature_columns[1][interval_sparse:]]

    print('Train data:', feature_columns)
    print('\nData shape:', np.array(X_train).shape)
 
    bottom_hidden_units = [256, 128, 64]
    bottom_output_dim = 64

    top_hidden_units = [128, 128, 64]
    top_output_dim = 1

    activation = 'relu'

    host_bottom_model = tf_host_bottom_graph(feat_host, bottom_hidden_units, bottom_output_dim, activation)
    guest_bottom_model = tf_guest_bottom_graph(feat_guest, bottom_hidden_units, bottom_output_dim, activation)
    top_model = tf_top_graph(k, w_reg, v_reg, top_hidden_units, top_output_dim, activation)
       
    # optimizer = optimizers.Adam(
    # learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    optimizer = optimizers.SGD(learning_rate=0.005, momentum=0.5)

    for i in range(500):
        with tf.GradientTape() as top_tape, tf.GradientTape() as host_bottom_tape, \
            tf.GradientTape() as guest_bottom_tape:

            host_bottom_output = host_bottom_model(X_train_host)
            guest_bottom_output = guest_bottom_model(X_train_guest)
            y_pre = top_model(host_bottom_output, guest_bottom_output)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))

        top_grad = top_tape.gradient(loss, top_model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(top_grad, top_model.variables))

        host_bottom_grad = host_bottom_tape.gradient(loss, host_bottom_model.variables)     
        optimizer.apply_gradients(grads_and_vars=zip(host_bottom_grad, host_bottom_model.variables))

        guest_bottom_grad = guest_bottom_tape.gradient(loss, guest_bottom_model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(guest_bottom_grad, guest_bottom_model.variables))

        test_host_bottom_output = host_bottom_model(X_test_host)
        test_guest_bottom_output = guest_bottom_model(X_test_guest)
        pre = top_model(test_host_bottom_output, test_guest_bottom_output)
        pre = [1 if x>0.5 else 0 for x in pre]
        auc = accuracy_score(y_test, pre)
        print('For the {}-th epoch, train loss: {}, test auc: {}'.format(i, loss.numpy(), auc))
