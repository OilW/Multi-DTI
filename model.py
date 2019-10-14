# encoding: utf-8
'''
@author: yuyou
@file: model.py
@time: 2019/7/15 19:24
@desc:
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras
from keras import backend as K
from keras.models import Model , Sequential
from keras.layers import Dense ,LSTM , Dropout , merge , Input , Embedding , Conv1D , GlobalMaxPooling1D
from keras.optimizers import Adam, SGD
from keras.losses import mse , binary_crossentropy
import numpy as np
from collections import OrderedDict
from Metrics import get_auroc , get_aupr , get_cindex , get_nce , get_ndcg , get_acc , get_precision , get_recall
from Input import shuffle_dataset , read_shuffled_data , shuffle_dataset_with_negative_sample , read_shuffled_data_with_negative_sample
from change_transformer import changed_transformer_model , build_combined_cos , build_combined_categorical
import tensorflow as tf

FOLDS = 5
Por = 0.6


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def DeepDTA(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot):
        deep_model = build_combined_categorical(
            XD_token_num=len(XD_all_train[0]), XP_token_num=len(XP_all_train[0]),
            len_drug=len_drug, len_prot=len_prot,
        )
        deep_model.compile(
            optimizer='Adam',
            loss='mse',
            metrics=[get_cindex]
        )
        print(deep_model.summary())

        deep_result = deep_model.fit(
            x=([XD_all_train ,XP_all_train]),
            y=np.array(Y_train),
            batch_size=256,
            epochs=200,
            validation_data=(([XD_all_test,XP_all_test]), np.array(Y_test)),
            shuffle=False)


def DeepDTA_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug, len_prot):
    deep_model = build_combined_categorical(
        XD_token_num=len(XD_all_train[0]), XP_token_num=len(XP_all_train[0]),
        len_drug=len_drug, len_prot=len_prot,
    )
    deep_model.compile(
        optimizer=Adam(lr = 0.00001),
        loss='binary_crossentropy',
        metrics=[keras.metrics.mse , keras.metrics.binary_crossentropy]
    )
    print(deep_model.summary())

    deep_result = deep_model.fit(
        x=([XD_all_train, XP_all_train]),
        y=np.array(Y_train),
        batch_size=64,
        epochs=200,
        validation_data=(([XD_all_test, XP_all_test]), np.array(Y_test)),
        shuffle=False)

def create_half_mse_bce(y_true, y_pred):
    por = Por
    return por*mse(y_true, y_pred) + (1-por)*binary_crossentropy(y_true, y_pred)

def DeepDTA_numerical_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug, len_prot):

    deep_model = build_combined_categorical(
        XD_token_num=len(XD_all_train[0]), XP_token_num=len(XP_all_train[0]),
        len_drug=len_drug, len_prot=len_prot,
    )
    deep_model.compile(
        optimizer=Adam(lr = 0.00001),
        loss=create_half_mse_bce,
        metrics=[get_cindex , keras.metrics.mse , keras.metrics.binary_crossentropy , keras.metrics.mae ]
    )
    print(deep_model.summary())

    for i in range(1000):
        print('Epoch:\t', i)
        deep_result = deep_model.fit(
            x=([XD_all_train, XP_all_train]),
            y=np.array(Y_train),
            batch_size=64,
            epochs=1,
            validation_data=(([XD_all_test, XP_all_test]), np.array(Y_test)),
            shuffle=False)

        Y_test_pred = deep_model.predict(x=([XD_all_test, XP_all_test]))

        np.savetxt('Y_test_dta.txt' , Y_test , fmt='%f')
        np.savetxt('Y_test_dta_pred.txt' , Y_test_pred , fmt='%f')

        x = np.loadtxt('Y_test_dta.txt')
        y = np.loadtxt('Y_test_dta_pred.txt')
        print(y)
        print('val_ndcg\t' , get_ndcg(x , y , k=100) , '\tval_aoc\t' , get_auroc(x , y) , '\tval_aupr\t' , get_aupr(x , y) , '\tval_precision\t' , get_precision(x , y) , '\tval_recall\t' , get_recall(x , y) , '\tval_acc\t' , get_acc(x , y))

def cnn_and_transformer(XD_encode_train, XD_decode_train, XD_all_train, XP_encode_train, XP_decode_train, XP_all_train, Y_train, XD_encode_test, XD_decode_test, XD_all_test, XP_encode_test, XP_decode_test, XP_all_test, Y_test, len_drug, len_prot, set_embed_dim):
        deep_model = changed_transformer_model_add_cnn(
            XD_token_num=len(XD_encode_train[0]), XP_token_num=len(XP_encode_train[0]),
            embed_dim=set_embed_dim,
            encoder_num=1,
            decoder_num=1,
            len_drug=len_drug,
            len_drug_all = np.shape(XD_all_train)[1] ,
            len_prot=len_prot,
            len_prot_all = np.shape(XP_all_train)[1] ,
            head_num=5,
            hidden_dim=120,
            attention_activation='relu',
            feed_forward_activation='relu',
            dropout_rate=0.5,
            XD_embed_weights=np.random.rand(len_drug, set_embed_dim),
            XP_embed_weights=np.random.rand(len_prot, set_embed_dim),
        )
        deep_model.compile(
            optimizer='Adam',
            loss='mean_squared_error',
            metrics=[get_cindex]
        )
        print(deep_model.summary())

        deep_result = deep_model.fit(
            x=([XD_encode_train , XD_decode_train , XP_encode_train , XP_decode_train ,XD_all_train ,XP_all_train]),
            y=np.array(Y_train),
            batch_size=256,
            epochs=200,
            validation_data=(([XD_encode_test, XD_decode_test , XP_encode_test , XP_decode_test , XD_all_test,XP_all_test]), np.array(Y_test)),
            shuffle=False )

def cnn_cos_cnn(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot):
    deep_model = build_combined_cos(
        XD_token_num=len(XD_all_train[0]), XP_token_num=len(XP_all_train[0]),
        len_drug=len_drug, len_prot=len_prot,
    )
    deep_model.compile(
        optimizer='Adam',
        loss='mse',
        metrics=[get_cindex , keras.metrics.mse , keras.metrics.binary_crossentropy]
    )
    print(deep_model.summary())

    '''
        '''
    deep_result = deep_model.fit(
        x=([XD_all_train, XP_all_train]),
        y=np.array(Y_train),
        batch_size=256,
        epochs=200,
        validation_data=(([XD_all_test, XP_all_test]), np.array(Y_test)),
        shuffle=False)


    Y_test_pred = deep_model.predict(x=([XD_all_test, XP_all_test]))

    np.savetxt('Y_test.txt' , Y_test , fmt='%f')
    np.savetxt('Y_test_pred.txt' , Y_test_pred , fmt='%f')

    x = np.loadtxt('Y_test.txt')
    length = np.shape(x)[0]
    y = np.loadtxt('Y_test_pred.txt')
    print(x)
    print(y)
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    with tf.Session() as sess:
        data_numpy = get_nce(x, y).eval()
        print(data_numpy)
        data_numpy /= length
        print(data_numpy)

def cnn_cos_cnn_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot):
    deep_model = build_combined_cos(
        XD_token_num=len(XD_all_train[0]), XP_token_num=len(XP_all_train[0]),
        len_drug=len_drug, len_prot=len_prot,
    )
    deep_model.compile(
        optimizer=Adam(lr = 0.00001),
        loss='binary_crossentropy',
        metrics=[keras.metrics.mse , keras.metrics.binary_crossentropy]
    )
    print(deep_model.summary())

    '''
        '''
    deep_result = deep_model.fit(
        x=([XD_all_train, XP_all_train]),
        y=np.array(Y_train),
        batch_size=64,
        epochs=200,
        validation_data=(([XD_all_test, XP_all_test]), np.array(Y_test)),
        shuffle=False)


    Y_test_pred = deep_model.predict(x=([XD_all_test, XP_all_test]))

    np.savetxt('Y_test.txt' , Y_test , fmt='%f')
    np.savetxt('Y_test_pred.txt' , Y_test_pred , fmt='%f')

    x = np.loadtxt('Y_test.txt')
    length = np.shape(x)[0]
    y = np.loadtxt('Y_test_pred.txt')
    print(x)
    print(y)
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    with tf.Session() as sess:
        data_numpy = get_nce(x, y).eval()
        print(data_numpy)
        data_numpy /= length
        print(data_numpy)


def cnn_cos_cnn_numerical_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot):
    deep_model = build_combined_cos(
        XD_token_num=len(XD_all_train[0]), XP_token_num=len(XP_all_train[0]),
        len_drug=len_drug, len_prot=len_prot,
    )
    por = Por
    print('por:' , por)
    deep_model.compile(
        optimizer=keras.optimizers.adam(lr=0.00001),
        loss=create_half_mse_bce,
        metrics=[get_cindex , keras.metrics.mse , keras.metrics.binary_crossentropy , keras.metrics.categorical_accuracy]
    )
    print(deep_model.summary())

    '''
        '''
    for i in range(300):
        print('Epoch:\t' , i)
        deep_result = deep_model.fit(
            x=([XD_all_train, XP_all_train]),
            y=np.array(Y_train),
            batch_size=64,
            epochs=1,
            validation_data=(([XD_all_test, XP_all_test]), np.array(Y_test)),
            shuffle=False)


        Y_test_pred = deep_model.predict(x=([XD_all_test, XP_all_test]))

        np.savetxt('Y_test.txt' , Y_test , fmt='%f')
        np.savetxt('Y_test_pred.txt' , Y_test_pred , fmt='%f')

        x = np.loadtxt('Y_test.txt')
        length = np.shape(x)[0]
        y = np.loadtxt('Y_test_pred.txt')

        print('val_ndcg\t' , get_ndcg(x , y) , 'val_aoc\t' , get_auroc(x , y) , 'val_aupr\t' , get_aupr(x , y))

    '''
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    with tf.Session() as sess:
        data_numpy = get_nce(x, y).eval()
        print(data_numpy)
        data_numpy /= length
        print(data_numpy)
    '''

def transformer_cos_transformer_numerical_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot):
    deep_model = build_combined_cos(
        XD_token_num=len(XD_all_train[0]), XP_token_num=len(XP_all_train[0]),
        len_drug=len_drug, len_prot=len_prot,
    )
    deep_model.compile(
        optimizer=keras.optimizers.adam(lr=0.00001),
        loss=create_half_mse_bce,
        metrics=[get_cindex , keras.metrics.mse , keras.metrics.binary_crossentropy , keras.metrics.categorical_accuracy]
    )
    print(deep_model.summary())

    '''
        '''
    for i in range(1000):
        deep_result = deep_model.fit(
            x=([XD_all_train, XP_all_train]),
            y=np.array(Y_train),
            batch_size=64,
            epochs=1,
            validation_data=(([XD_all_test, XP_all_test]), np.array(Y_test)),
            shuffle=False)


        Y_test_pred = deep_model.predict(x=([XD_all_test, XP_all_test]))

        np.savetxt('Y_test.txt' , Y_test , fmt='%f')
        np.savetxt('Y_test_pred.txt' , Y_test_pred , fmt='%f')

        x = np.loadtxt('Y_test.txt')
        length = np.shape(x)[0]
        y = np.loadtxt('Y_test_pred.txt')

        print('val_ndcg\t' , get_ndcg(x , y) , 'val_aoc\t' , get_auroc(x , y) , 'val_aupr\t' , get_aupr(x , y))

    '''
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    with tf.Session() as sess:
        data_numpy = get_nce(x, y).eval()
        print(data_numpy)
        data_numpy /= length
        print(data_numpy)
    '''

def experiment_numerical(data_code , embedding_code , model_code):

    for selected_test_data in range(1 , 2):
        '''
        XD_encode_train = np.random.rand(10 , 90)
        XD_decode_train = np.random.rand(10 , 90)
        XD_all_train = np.random.rand(10 , 180)
        XP_encode_train = np.random.rand(10 , 80)
        XP_decode_train = np.random.rand(10 , 80)
        XP_all_train = np.random.rand(10 , 160)
        Y_train = np.random.rand(10 )
        XD_encode_test = np.random.rand(6 , 90)
        XD_decode_test = np.random.rand(6 , 90)
        XD_all_test = np.random.rand(6 , 180)
        XP_encode_test = np.random.rand(6 , 80)
        XP_decode_test = np.random.rand(6 , 80)
        XP_all_test = np.random.rand(6 , 160)
        Y_test = np.random.rand(6)
        '''
        print('selected_test_data:' , selected_test_data)
        XD_encode_train, XD_decode_train, XD_all_train, XP_encode_train, XP_decode_train, XP_all_train, Y_train, XD_encode_test, XD_decode_test, XD_all_test, XP_encode_test, XP_decode_test, XP_all_test, Y_test= read_shuffled_data(
            data_code=data_code ,
            embedding_code=1 ,
            fold_num = selected_test_data,
            folds = FOLDS)


        x = [XD_encode_train , XD_decode_train , XP_encode_train , XP_decode_train , Y_train , XD_encode_test , XD_decode_test , XP_encode_test , XP_decode_test , Y_test]

        len_drug = max(np.max(XD_encode_train) , np.max(XD_decode_train) , np.max(XD_encode_test) , np.max(XD_decode_test))
        len_prot = max(np.max(XP_encode_train) , np.max(XP_decode_train) , np.max(XP_encode_test) , np.max(XP_decode_test))
        len_drug += 5
        len_prot += 5
        for i in x:
            print(np.shape(i))
            print(np.max(i) , np.min(i))

        XD_encode_train =XD_encode_train.astype(np.float)
        XD_encode_train[XD_encode_train==np.min(XD_encode_train)] = 1e-6
        XD_decode_train =XD_decode_train.astype(np.float)
        XD_decode_train[XD_decode_train==np.min(XD_decode_train)] = 1e-6
        XP_encode_train =XP_encode_train.astype(np.float)
        XP_encode_train[XP_encode_train==np.min(XP_encode_train)] = 1e-6
        XP_decode_train =XP_decode_train.astype(np.float)
        XP_decode_train[XP_decode_train==np.min(XP_decode_train)] = 1e-6
        XD_encode_test =XD_encode_test.astype(np.float)
        XD_encode_test[XD_encode_test==np.min(XD_encode_test)] = 1e-6
        XD_decode_test =XD_decode_test.astype(np.float)
        XD_decode_test[XD_decode_test==np.min(XD_decode_test)] = 1e-6
        XP_encode_test =XP_encode_test.astype(np.float)
        XP_encode_test[XP_encode_test==np.min(XP_encode_test)] = 1e-6
        XP_decode_test = XP_decode_test.astype(np.float)
        XP_decode_test[XP_decode_test==np.min(XP_decode_test)] = 1e-6



        print(np.average(Y_train) , np.max(Y_train) , np.min(Y_train))
        print(np.average(Y_test) , np.max(Y_test) , np.min(Y_test))

        set_embed_dim = 30
        '''
        '''
        #cnn_and_transformer(XD_encode_train, XD_decode_train, XD_all_train, XP_encode_train, XP_decode_train,XP_all_train, Y_train, XD_encode_test, XD_decode_test, XD_all_test, XP_encode_test, XP_decode_test, XP_all_test, Y_test, len_drug, len_prot, set_embed_dim)
        DeepDTA(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot)

        #cnn_cos_cnn(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot)

def experiment_categorical(data_code):
    for fold_index in range(FOLDS):
        XD_encode_train, XD_decode_train, XD_all_train, XP_encode_train, XP_decode_train, XP_all_train, Y_train, XD_encode_test, XD_decode_test, XD_all_test, XP_encode_test, XP_decode_test, XP_all_test, Y_test= read_shuffled_data(
                data_code=data_code ,
                embedding_code=1 ,
                fold_num = fold_index ,
                folds = FOLDS)

        print('train:' , np.average(Y_train) , np.max(Y_train) , np.min(Y_train) , Y_train[0])
        print('test:' , np.average(Y_test) , np.max(Y_test) , np.min(Y_test) , Y_test[0])

        x = [XD_encode_train , XD_decode_train , XP_encode_train , XP_decode_train , Y_train , XD_encode_test , XD_decode_test , XP_encode_test , XP_decode_test , Y_test]

        len_drug = max(np.max(XD_encode_train) , np.max(XD_decode_train) , np.max(XD_encode_test) , np.max(XD_decode_test))
        len_prot = max(np.max(XP_encode_train) , np.max(XP_decode_train) , np.max(XP_encode_test) , np.max(XP_decode_test))
        len_drug += 5
        len_prot += 5

        XD_encode_train =XD_encode_train.astype(np.float)
        XD_encode_train[XD_encode_train==np.min(XD_encode_train)] = 1e-6
        XD_decode_train =XD_decode_train.astype(np.float)
        XD_decode_train[XD_decode_train==np.min(XD_decode_train)] = 1e-6
        XP_encode_train =XP_encode_train.astype(np.float)
        XP_encode_train[XP_encode_train==np.min(XP_encode_train)] = 1e-6
        XP_decode_train =XP_decode_train.astype(np.float)
        XP_decode_train[XP_decode_train==np.min(XP_decode_train)] = 1e-6
        XD_encode_test =XD_encode_test.astype(np.float)
        XD_encode_test[XD_encode_test==np.min(XD_encode_test)] = 1e-6
        XD_decode_test =XD_decode_test.astype(np.float)
        XD_decode_test[XD_decode_test==np.min(XD_decode_test)] = 1e-6
        XP_encode_test =XP_encode_test.astype(np.float)
        XP_encode_test[XP_encode_test==np.min(XP_encode_test)] = 1e-6
        XP_decode_test = XP_decode_test.astype(np.float)
        XP_decode_test[XP_decode_test==np.min(XP_decode_test)] = 1e-6

        for i in x:
            print(np.shape(i))
            print(np.max(i) , np.min(i))

        set_embed_dim = 30
        '''
        '''
        print(Y_test)
        y_true = np.where(Y_test >= 0.001, 1, 0)
        print(y_true)
        #cnn_and_transformer(XD_encode_train, XD_decode_train, XD_all_train, XP_encode_train, XP_decode_train,XP_all_train, Y_train, XD_encode_test, XD_decode_test, XD_all_test, XP_encode_test, XP_decode_test, XP_all_test, Y_test, len_drug, len_prot, set_embed_dim)
        #DeepDTA(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot)

        #DeepDTA_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug, len_prot)

        print('Cross validation number:' , fold_index)
        cnn_cos_cnn_numerical_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug, len_prot)
        #DeepDTA_numerical_categorical(XD_all_train, XP_all_train, Y_train, XD_all_test, XP_all_test, Y_test, len_drug , len_prot)

if __name__ == "__main__":
    #data_code={1:davis ; 2:KIBA ; 3:DTINet}
    data_code = 2

    #embedding_code={1:pubchem+SW ; 2:one-hot ; 3:word embedding} ???maybe useless
    embedding_code = 1

    #model_code = {1:baseline , 2:changed_transformer}
    model_code = 2

    '''
    shuffle_dataset(data_code , folds=5)
    experiment_numerical(data_code , embedding_code , model_code)
    '''

    shuffle_dataset_with_negative_sample(data_code , folds=FOLDS)
    experiment_categorical(data_code=data_code)
