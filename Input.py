# encoding: utf-8
'''
@author: yuyou
@file: Input.py
@time: 2019/7/16 16:45
@desc:
'''


import numpy as np
import os
import random
import tensorflow as tf


NUM_FILTERS = 32
FILTER_LENGTH1 = 4
FILTER_LENGTH2 = 8

CHAR_SMI_SET = {'<START>': 1,'<END>': 2,'<PAD>': 0,}
with open(r'drugs_chars.txt' , 'r') as f:
    for line in f.readlines():
        CHAR_SMI_SET[line.strip()] = len(CHAR_SMI_SET)


CHAR_PROT_SET = {'<START>': 1,'<END>': 2, '<PAD>': 0,}
'''
with open(r'data\all_embeddings\prots_chars.txt' , 'r') as f:
    for line in f.readlines():
        CHAR_PROT_SET[line.strip()] = len(CHAR_PROT_SET)
'''





def shuffle_dataset(data_code , folds = 5):
    paths ={1:r"data/davis/", 2:r"data/KIBA/" , 3:r"data/DTINet/"}
    data_path = paths[data_code]
    print('shuffle dataset! the data_path:' , data_path)

    '''
    with open(data_path + 'positive_pairs.txt' , 'r') as f:
        train_all = f.readlines()
    '''
    train_all = []
    with open(data_path + 'positive_pairs.txt' , 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if len(item)==3:
                if len(item[1])<1000 and len(item[0]) < 100:
                    train_all.append(line)


    random.Random(1000000).shuffle(train_all)

    n = int(len(train_all) / folds)+1
    train_all = [train_all[i:i + n] for i in range(0, len(train_all), n)]

    for file in os.listdir('temp_shuffle'):
        os.remove('temp_shuffle/' + file)
    for i in range(folds):
        with open('temp_shuffle/train_part_' + str(i) + '.txt' , 'w') as f:
            for j in train_all[i]:
                f.write(j)


def shuffle_dataset_with_negative_sample(data_code , folds = 5 , sample_rate = 5):
    paths ={1:r"data/davis/", 2:r"data/KIBA/" , 3:r"data/DTINet/"}
    data_path = paths[data_code]
    print('shuffle dataset! the data_path:' , data_path)

    train_all = []
    max_label = 0
    with open(data_path + 'positive_pairs.txt' , 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if len(item)==3:
                if len(item[1])<1000 and len(item[0]) < 100:
                    item[2] = float(item[2])
                    if item[2] > max_label:
                        max_label = item[2]
                    #line = item[0] + '\t' + item[1] + '\t1\n'
                    train_all.append(item)
    min_label = int(max_label/2)-0.001
    temp = []
    for item in train_all:
        if item[2] > min_label:
            temp.append(item[0] + '\t' + item[1] + '\t' + str((item[2] - min_label) / (max_label - min_label)) + '\n')
    train_all = temp

    random.Random(1000000).shuffle(train_all)

    length_positive = len(train_all)
    length_negative = sample_rate * length_positive

    n = int(length_positive / folds)+1
    train_all = [train_all[i:i + n] for i in range(0, length_positive, n)]

    for file in os.listdir('temp_shuffle'):
        os.remove('temp_shuffle/' + file)
    for i in range(folds):
        with open('temp_shuffle/train_part_' + str(i) + '.txt' , 'w') as f:
            for j in train_all[i]:
                f.write(j)

    train_all = []
    with open(data_path + 'negative_pairs.txt', 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if len(item) == 3:
                if len(item[1]) < 1000 and len(item[0]) < 100:
                    line = item[0] + '\t' + item[1] + '\t0\n'
                    train_all.append(line)

    random.Random(1000000).shuffle(train_all)
    if len(train_all)< length_negative:
        length_negative = len(train_all)

    n = int(length_negative / folds) + 1
    train_all = [train_all[i:i + n] for i in range(0, length_negative, n)]

    for i in range(folds):
        with open('temp_shuffle/train_part_' + str(i) + '.txt', 'a') as f:
            for j in train_all[i]:
                f.write(j)


def n_non_overlapping_gram(str , n = 2):
    str_list = []
    temp = ''
    for char in range(len(str)):
        temp += str[char]
        if char % n == n-1:
            str_list.append(temp)
            temp = ''
    str_list.append(temp)
    return str_list

def data_single(XD_train , XP_train , Y_train , XD_test , XP_test , Y_test , max_drug_len , max_prot_len):
    half_max_drug_len = int(max_drug_len/2) + 1
    half_max_prot_len = int(max_prot_len/2) + 1

    XD_encode_train = []
    XD_decode_train = []
    XD_all_train = []
    for i in range(len(XD_train)):
        XD_train[i] = list(XD_train[i])
        encode_tokens, decode_tokens = XD_train[i][:int(len(XD_train[i])/2)], XD_train[i][int(len(XD_train[i])/2):]
        encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (half_max_drug_len - len(encode_tokens))
        decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (half_max_drug_len - len(decode_tokens))
        all_tokens = XD_train[i] + ['<PAD>'] * (max_drug_len - len(XD_train[i]))
        encode_tokens = np.array(list(map(lambda x: CHAR_SMI_SET[x], encode_tokens)))
        decode_tokens = np.array(list(map(lambda x: CHAR_SMI_SET[x], decode_tokens)))
        all_tokens = np.array(list(map(lambda x: CHAR_SMI_SET[x], all_tokens)))
        XD_encode_train.append(encode_tokens)
        XD_decode_train.append(decode_tokens)
        XD_all_train.append(all_tokens)

    XD_encode_train = np.array(XD_encode_train)
    XD_decode_train = np.array(XD_decode_train)
    XD_all_train = np.array(XD_all_train)

    XP_encode_train = []
    XP_decode_train = []
    XP_all_train = []
    for i in range(len(XP_train)):
        XP_train[i] = n_non_overlapping_gram(XP_train[i])
        encode_tokens, decode_tokens = XP_train[i][:int(len(XP_train[i])/2)], XP_train[i][int(len(XP_train[i])/2):]
        encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (half_max_prot_len - len(encode_tokens))
        decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (half_max_prot_len - len(decode_tokens))
        all_tokens = XP_train[i] + ['<PAD>'] * (max_prot_len - len(XP_train[i]))
        encode_tokens = np.array(list(map(lambda x: CHAR_PROT_SET[x], encode_tokens)))
        decode_tokens = np.array(list(map(lambda x: CHAR_PROT_SET[x], decode_tokens)))
        all_tokens = np.array(list(map(lambda x: CHAR_PROT_SET[x], all_tokens)))
        XP_encode_train.append(encode_tokens)
        XP_decode_train.append(decode_tokens)
        XP_all_train.append(all_tokens)
    XP_encode_train = np.array(XP_encode_train)
    XP_decode_train = np.array(XP_decode_train)
    XP_all_train = np.array(XP_all_train)

    Y_train = np.array(Y_train)

    XD_encode_test = []
    XD_decode_test = []
    XD_all_test = []
    for i in range(len(XD_test)):
        XD_test[i] = list(XD_test[i])
        encode_tokens, decode_tokens = XD_test[i][:int(len(XD_test[i])/2)], XD_test[i][int(len(XD_test[i])/2):]
        encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (half_max_drug_len - len(encode_tokens))
        decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (half_max_drug_len - len(decode_tokens))
        all_tokens = XD_test[i] + ['<PAD>'] * (max_drug_len - len(XD_test[i]))
        encode_tokens = np.array(list(map(lambda x: CHAR_SMI_SET[x], encode_tokens)))
        decode_tokens = np.array(list(map(lambda x: CHAR_SMI_SET[x], decode_tokens)))
        all_tokens = np.array(list(map(lambda x: CHAR_SMI_SET[x], all_tokens)))
        XD_encode_test.append(encode_tokens)
        XD_decode_test.append(decode_tokens)
        XD_all_test.append(all_tokens)
    XD_encode_test = np.array(XD_encode_test)
    XD_decode_test = np.array(XD_decode_test)
    XD_all_test = np.array(XD_all_test)

    XP_encode_test = []
    XP_decode_test = []
    XP_all_test = []
    for i in range(len(XP_test)):
        XP_test[i] = n_non_overlapping_gram(XP_test[i])
        encode_tokens, decode_tokens = XP_test[i][:int(len(XP_test[i])/2)], XP_test[i][int(len(XP_test[i])/2):]
        encode_tokens = ['<START>'] + encode_tokens + ['<END>'] + ['<PAD>'] * (half_max_prot_len - len(encode_tokens))
        decode_tokens = ['<START>'] + decode_tokens + ['<END>'] + ['<PAD>'] * (half_max_prot_len - len(decode_tokens))
        all_tokens = XP_test[i] + ['<PAD>'] * (max_prot_len - len(XP_test[i]))
        encode_tokens = np.array(list(map(lambda x: CHAR_PROT_SET[x], encode_tokens)))
        decode_tokens = np.array(list(map(lambda x: CHAR_PROT_SET[x], decode_tokens)))
        all_tokens = np.array(list(map(lambda x: CHAR_PROT_SET[x], all_tokens)))
        XP_encode_test.append(encode_tokens)
        XP_decode_test.append(decode_tokens)
        XP_all_test.append(all_tokens)
    XP_encode_test = np.array(XP_encode_test)
    XP_decode_test = np.array(XP_decode_test)
    XP_all_test = np.array(XP_all_test)

    Y_test = np.array(Y_test)

    max_label = max(np.max(Y_train), np.max(Y_test))
    min_label = 0.5 * max_label
    Y_train = np.where(Y_train >= 1e-6, (Y_train - min_label) / (max_label - min_label), 0)
    Y_train = np.where(Y_train >= 0, Y_train , 1e-5)
    Y_test = np.where(Y_test >= 1e-6, (Y_test - min_label) / (max_label - min_label), 0)
    Y_test = np.where(Y_test >= 0, Y_test , 1e-5)

    return XD_encode_train , XD_decode_train , XD_all_train , XP_encode_train , XP_decode_train , XP_all_train, Y_train , XD_encode_test , XD_decode_test , XD_all_test , XP_encode_test , XP_decode_test , XP_all_test , Y_test

def data_all(XD_train , XP_train , Y_train , XD_test , XP_test , Y_test , max_drug_len , max_prot_len):
    for i in range(len(XD_train)):
        temp = np.zeros(max_drug_len)
        for j in range(len(XD_train[i])):
            temp[j] = CHAR_SMI_SET[XD_train[i][j]]
        XD_train[i] = temp
    XD_train = np.asarray(XD_train)
    for i in range(len(XP_train)):
        temp = np.zeros(max_prot_len)
        for j in range(len(XP_train[i])):
            temp[j] = CHAR_PROT_SET[XP_train[i][j]]
        XP_train[i] = temp
    XP_train = np.asarray(XP_train)
    Y_train = np.asarray(Y_train)
    for i in range(len(XD_test)):
        temp = np.zeros(max_drug_len)
        for j in range(len(XD_test[i])):
            temp[j] = CHAR_SMI_SET[XD_test[i][j]]
        XD_test[i] = temp
    XD_test = np.asarray(XD_test)
    for i in range(len(XP_test)):
        temp = np.zeros(max_prot_len)
        for j in range(len(XP_test[i])):
            temp[j] = CHAR_PROT_SET[XP_test[i][j]]
        XP_test[i] = temp
    XP_test = np.asarray(XP_test)
    Y_test = np.asarray(Y_test)

    max_label = max(np.max(Y_train), np.max(Y_test))
    min_label = min(np.min(Y_train), np.min(Y_test)) - 0.1
    Y_train = (Y_train - min_label) / (max_label - min_label)
    Y_test = (Y_test - min_label) / (max_label - min_label)

    return XD_train , XP_train , Y_train , XD_test , XP_test , Y_test

def read_shuffled_data(data_code , embedding_code , fold_num , folds = 5):
    paths ={1:r"data/davis/", 2:r"data/KIBA/" , 3:r"data/DTINet/"}
    data_path = paths[data_code]
    print('data_path:' , data_path)

    XD_train = []
    XP_train = []
    Y_train = []
    XD_test = []
    XP_test = []
    Y_test = []
    max_drug_len = 0
    max_prot_len = 0
    for fold in range(folds):
        if fold != fold_num:
            with open('temp_shuffle/train_part_' + str(fold) + '.txt' , 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    if len(line) == 3:
                        XD_train.append(line[0])
                        if len(line[0]) > max_drug_len:
                            max_drug_len = len(line[0])
                        XP_train.append(line[1])
                        temp_p = n_non_overlapping_gram(line[1])
                        for item in temp_p:
                            if item not in CHAR_PROT_SET.keys():
                                CHAR_PROT_SET[item] = len(CHAR_PROT_SET)
                        if len(temp_p) > max_prot_len:
                            max_prot_len = len(temp_p)
                        Y_train.append(float(line[2]))
        else:
            with open('temp_shuffle/train_part_' + str(fold) + '.txt' , 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    if len(line) == 3:
                        XD_test.append(line[0])
                        if len(line[0]) > max_drug_len:
                            max_drug_len = len(line[0])
                        XP_test.append(line[1])
                        temp_p = n_non_overlapping_gram(line[1])
                        for item in temp_p:
                            if item not in CHAR_PROT_SET.keys():
                                CHAR_PROT_SET[item] = len(CHAR_PROT_SET)
                        if len(temp_p) > max_prot_len:
                            max_prot_len = len(temp_p)
                        Y_test.append(float(line[2]))

    XD_encode_train, XD_decode_train, XD_all_train, XP_encode_train, XP_decode_train, XP_all_train, Y_train, XD_encode_test, XD_decode_test, XD_all_test, XP_encode_test, XP_decode_test, XP_all_test, Y_test = data_single(XD_train , XP_train , Y_train , XD_test , XP_test , Y_test , max_drug_len , max_prot_len)
    print(fold_num , 'th fold:read dataset complete!')
    print('number of training set:' , len(XD_encode_train))
    print('number of testing set:' , len(XD_encode_test))
    return  XD_encode_train , XD_decode_train , XD_all_train , XP_encode_train , XP_decode_train , XP_all_train, Y_train , XD_encode_test , XD_decode_test , XD_all_test , XP_encode_test , XP_decode_test , XP_all_test , Y_test

def read_shuffled_data_with_negative_sample(data_code , embedding_code , fold_num , folds = 5):
    paths ={1:r"data/davis/", 2:r"data/KIBA/" , 3:r"data/DTINet/"}
    data_path = paths[data_code]
    print('data_path:' , data_path)

    XD_train = []
    XP_train = []
    Y_train = []
    XD_test = []
    XP_test = []
    Y_test = []
    max_drug_len = 0
    max_prot_len = 0
    for fold in range(folds):
        if fold != fold_num:
            with open('temp_shuffle/train_part_' + str(fold) + '.txt' , 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    if len(line) == 3:
                        XD_train.append(line[0])
                        if len(line[0]) > max_drug_len:
                            max_drug_len = len(line[0])
                        XP_train.append(line[1])
                        temp_p = n_non_overlapping_gram(line[1])
                        for item in temp_p:
                            if item not in CHAR_PROT_SET.keys():
                                CHAR_PROT_SET[item] = len(CHAR_PROT_SET)
                        if len(temp_p) > max_prot_len:
                            max_prot_len = len(temp_p)
                        Y_train.append(int(line[2]))
        else:
            with open('temp_shuffle/train_part_' + str(fold) + '.txt' , 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    if len(line) == 3:
                        XD_test.append(line[0])
                        if len(line[0]) > max_drug_len:
                            max_drug_len = len(line[0])
                        XP_test.append(line[1])
                        temp_p = n_non_overlapping_gram(line[1])
                        for item in temp_p:
                            if item not in CHAR_PROT_SET.keys():
                                CHAR_PROT_SET[item] = len(CHAR_PROT_SET)
                        if len(temp_p) > max_prot_len:
                            max_prot_len = len(temp_p)
                        Y_test.append(int(line[2]))

    XD_encode_train, XD_decode_train, XD_all_train, XP_encode_train, XP_decode_train, XP_all_train, Y_train, XD_encode_test, XD_decode_test, XD_all_test, XP_encode_test, XP_decode_test, XP_all_test, Y_test = data_single(XD_train , XP_train , Y_train , XD_test , XP_test , Y_test , max_drug_len , max_prot_len)
    print(fold_num , 'th fold:read dataset complete!')
    print('number of training set:' , len(XD_encode_train))
    print('number of testing set:' , len(XD_encode_test))
    return  XD_encode_train , XD_decode_train , XD_all_train , XP_encode_train , XP_decode_train , XP_all_train, Y_train , XD_encode_test , XD_decode_test , XD_all_test , XP_encode_test , XP_decode_test , XP_all_test , Y_test


def create_CPI_data(data_code , folds = 5 , sample_rate = 5):
    paths ={1:r"data/davis/", 2:r"data/KIBA/" , 3:r"data/DTINet/"}
    data_path = paths[data_code]
    print('create CPI dataset! the data path:' , data_path)

    train_all = []
    max_label = 0
    with open(data_path + 'positive_pairs.txt' , 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if len(item)==3:
                if len(item[1])<1000 and len(item[0]) < 100:
                    item[2] = float(item[2])
                    if item[2] > max_label:
                        max_label = item[2]
                    #line = item[0] + '\t' + item[1] + '\t1\n'
                    train_all.append(item)
    min_label = int(max_label/2)-0.001



    temp = []
    for item in train_all:
        if item[2] > min_label:
            temp.append(item[0] + '\t' + item[1] + '\t' + str((item[2] - min_label) / (max_label - min_label)) + '\n')
    train_all = temp
    print(len(train_all))


    random.Random(1000000).shuffle(train_all)

    length_positive = len(train_all)
    length_negative = sample_rate * length_positive

    with open('CPI_data.txt' , 'w') as f:
        for line in train_all:
            f.write(line)

    train_all = []
    with open(data_path + 'negative_pairs.txt', 'r') as f:
        for line in f.readlines():
            item = line.strip().split()
            if len(item) == 3:
                if len(item[1]) < 1000 and len(item[0]) < 100:
                    line = item[0] + '\t' + item[1] + '\t0\n'
                    train_all.append(line)

    print(len(train_all))
    random.Random(1000000).shuffle(train_all)

    if len(train_all)< length_negative:
        length_negative = len(train_all)

    with open('CPI_data.txt' , 'a') as f:
        for line in range(length_negative):
            f.write(train_all[line])



