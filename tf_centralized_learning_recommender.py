'''
Centralized training
'''
from deepFM_model import DeepFM
from utils import create_criteo_dataset_dfm

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    file_path = './dataset/criteo.txt'
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset_dfm(file_path, data_size=50000, test_size=0.2)

    k = 10
    w_reg = 1e-4
    v_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'

    model = DeepFM(feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation)
    optimizer = optimizers.SGD(0.01)
    #optimizer = optimizers.SGD(learning_rate=0.005, momentum=0.5)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    for i in range(500):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

        #evaluate
        pre = model(X_test)
        pre = [1 if x>0.5 else 0 for x in pre]
        auc = accuracy_score(y_test, pre)
        print('For the {}-th epoch, train loss: {}, test auc: {}'.format(i, loss.numpy(), auc))
