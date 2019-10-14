# encoding: utf-8
'''
@author: yuyou
@file: temp.py
@time: 2019/7/21 22:04
@desc:
'''
import numpy as np
from keras_transformer import get_model
from contextlib import redirect_stdout
from pprint import pprint
import numpy as np
from keras_pos_embd import TrigPosEmbedding , PositionEmbedding
from keras_embed_sim import EmbeddingRet, EmbeddingSim
from keras_transformer.backend import keras
import keras_transformer
from keras import Model,regularizers
from keras.layers import Dense , Dropout , Input , Embedding , Conv1D , GlobalMaxPooling1D , Multiply
from keras import backend as K
from keras.engine.topology import Layer
from Metrics import get_cindex
import tensorflow as tf


class MyFlatten(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MyFlatten, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask==None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))


def build_combined_categorical(XD_token_num, XP_token_num, len_drug, len_prot, NUM_FILTERS=32, FILTER_LENGTH1=4, FILTER_LENGTH2=8):
    XDinput = Input(shape=(XD_token_num,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(XP_token_num,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=len_drug, output_dim=128, input_length=XD_token_num)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=len_prot, output_dim=128, input_length=XP_token_num)(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein],
                                                  axis=-1)  # merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2)  # OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    return interactionModel

def build_combined_cos(XD_token_num, XP_token_num, len_drug, len_prot, NUM_FILTERS=32, FILTER_LENGTH1=4, FILTER_LENGTH2=8):
    XDinput = Input(shape=(XD_token_num,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(XP_token_num,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=len_drug, output_dim=128, input_length=XD_token_num)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=len_prot, output_dim=128, input_length=XP_token_num)(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    #encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein],axis=-1)  # merge.Add()([encode_smiles, encode_protein])
    encode_interaction = keras.layers.Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1]), axis=1, keep_dims=False) / (tf.sqrt(tf.reduce_sum(tf.square(x[0]), axis=1))* tf.sqrt(tf.reduce_sum(tf.square(x[1]), axis=1))))([encode_smiles , encode_protein])
    encode_interaction = MyFlatten()(encode_interaction)
    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[encode_interaction])

    return interactionModel


def build_cnn_concat_cnn_attention(XD_token_num, XP_token_num, len_drug, len_prot, NUM_FILTERS=32, FILTER_LENGTH1=4, FILTER_LENGTH2=8):
    XDinput = Input(shape=(XD_token_num,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(XP_token_num,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=len_drug, output_dim=128, input_length=XD_token_num)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=len_prot, output_dim=128, input_length=XP_token_num)(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_smiles_probs = Dense(NUM_FILTERS * 3, activation='softmax', name='smiles_attention_vec')(encode_smiles)
    encode_protein_probs = Dense(NUM_FILTERS * 3, activation='softmax', name='protein_attention_vec')(encode_protein)
    encode_smiles_mul = Multiply()([encode_smiles , encode_protein_probs])
    encode_protein_mul = Multiply()([encode_protein , encode_smiles_probs])

    encode_interaction = keras.layers.concatenate([encode_smiles_mul, encode_protein_mul],axis=-1)  # merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2)  # OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    return interactionModel

def build_cnn_cos_cnn_attention(XD_token_num, XP_token_num, len_drug, len_prot, NUM_FILTERS=32, FILTER_LENGTH1=4, FILTER_LENGTH2=8):
    XDinput = Input(shape=(XD_token_num,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(XP_token_num,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=len_drug, output_dim=128, input_length=XD_token_num)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)
    #encode_smiles = Dense(NUM_FILTERS * 3, activation='relu' , kernel_regularizer=regularizers.l1(0.01))(encode_smiles)

    encode_protein = Embedding(input_dim=len_prot, output_dim=128, input_length=XP_token_num)(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)
    #encode_protein = Dense(NUM_FILTERS * 3, activation='relu' , kernel_regularizer=regularizers.l1(0.01))(encode_protein)

    encode_smiles_probs = Dense(NUM_FILTERS * 3, activation='softmax', name='smiles_attention_vec')(encode_smiles)
    encode_protein_probs = Dense(NUM_FILTERS * 3, activation='softmax', name='protein_attention_vec')(encode_protein)
    encode_smiles_mul = Multiply()([encode_smiles , encode_protein_probs])
    encode_protein_mul = Multiply()([encode_protein , encode_smiles_probs])
    encode_smiles_mul = keras.layers.Lambda(lambda x: tf.clip_by_value(x, 1e-6 , 1))(encode_smiles_mul)
    encode_protein_mul = keras.layers.Lambda(lambda x: tf.clip_by_value(x, 1e-6 , 1))(encode_protein_mul)


    '''
    encode_smiles_mul = Dropout(0.5)(encode_smiles_mul)
    encode_protein_mul = Dropout(0.5)(encode_protein_mul)
    encode_smiles_mul = Dense(NUM_FILTERS * 3, activation='softmax')(encode_smiles_mul)
    encode_protein_mul = Dense(NUM_FILTERS * 3, activation='softmax')(encode_protein_mul)
    '''

    encode_interaction = keras.layers.Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1]), axis=1, keep_dims=False) / (tf.sqrt(tf.reduce_sum(tf.square(x[0]), axis=1))* tf.sqrt(tf.reduce_sum(tf.square(x[1]), axis=1))))([encode_smiles_mul , encode_protein_mul])
    #encode_interaction = keras.layers.Lambda(lambda x: tf.clip_by_value(1e-6 , 1))(encode_interaction)
    encode_interaction = MyFlatten()(encode_interaction)
    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[encode_interaction])

    return interactionModel


if __name__ == "__main__":
    print(1)

