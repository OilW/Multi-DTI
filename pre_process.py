# encoding: utf-8
'''
@author: yuyou
@file: pre_process.py
@time: 2019/7/19 19:49
@desc:
'''

import json
import numpy as np
import tensorflow as tf
from Input import n_non_overlapping_gram
from gensim.models import word2vec

def Davis():
    data_path = r"data/davis/"
    XD = json.load(open(data_path+"drugs.txt"))
    XD = list(XD.values())
    XP = json.load(open(data_path+"proteins.txt"))
    XP = list(XP.values())
    Y = np.loadtxt(data_path + 'mat_drug_protein.txt')
    print(Y[0][0] , Y[0][1] , Y[0][2] , Y[1][0] )
    Y = -(np.log10(Y / (10 ** 9)))
    print(Y[0][0] , Y[0][1] , Y[0][2] , Y[1][0] )
    print(len(XD) , len(XP) , np.shape(Y))
    with open(data_path + 'positive_pairs.txt' , 'w') as fpositive:
        with open(data_path + 'negative_pairs.txt' , 'w') as fnegative:
            nans = np.isnan(Y)
            print(np.sum(nans))
            for d in range(len(XD)):
                for p in range(len(XP)):
                    if nans[d][p]:
                        fnegative.write(XD[d] + '\t' + XP[p] + '\tnan\n')
                    else:
                        fpositive.write(XD[d] + '\t' + XP[p] + '\t' + str(Y[d][p]) + '\n')

def KIBA():

    data_path = r"data/KIBA/"
    XD = json.load(open(data_path+"drugs.txt"))
    XD = list(XD.values())
    XP = json.load(open(data_path+"proteins.txt"))
    XP = list(XP.values())

    Y = np.loadtxt(data_path + 'mat_drug_protein.txt')

    print(len(XD) , len(XP) , np.shape(Y))
    with open(data_path + 'positive_pairs.txt' , 'w') as fpositive:
        with open(data_path + 'negative_pairs.txt' , 'w') as fnegative:
            nans = np.isnan(Y)
            print(np.sum(nans))
            for d in range(len(XD)):
                for p in range(len(XP)):
                    if nans[d][p]:
                        fnegative.write(XD[d] + '\t' + XP[p] + '\tnan\n')
                    else:
                        fpositive.write(XD[d] + '\t' + XP[p] + '\t' + str(Y[d][p]) + '\n')

def DTINet():

    data_path = r"data/DTINet/"
    XD = json.load(open(data_path+"drugs.txt"))
    XD = list(XD.values())
    XP = json.load(open(data_path+"proteins.txt"))
    XP = list(XP.values())

    Y = np.loadtxt(data_path + 'mat_drug_protein.txt')

    print(len(XD) , len(XP) , np.shape(Y))
    with open(data_path + 'positive_pairs.txt' , 'w') as fpositive:
        with open(data_path + 'negative_pairs.txt' , 'w') as fnegative:
            nans = np.isnan(Y)
            print(np.sum(nans))
            for d in range(len(XD)):
                for p in range(len(XP)):
                    if nans[d][p]:
                        fnegative.write(XD[d] + '\t' + XP[p] + '\tnan\n')
                    else:
                        fpositive.write(XD[d] + '\t' + XP[p] + '\t' + str(Y[d][p]) + '\n')

def create_pairs():
    Davis()
    #KIBA()
    #DTINet()

def embedding_protein():
    '''
    XP1 = json.load(open("data/Davis/proteins.txt"))
    XP = list(XP1.values())
    XP2 = json.load(open("data/KIBA/proteins.txt"))
    XP.extend(list(XP2.values()))
    XP3 = json.load(open("data/DTINet/proteins.txt"))
    XP.extend(list(XP3.values()))
    for i in range(len(XP)):
        XP[i] = n_non_overlapping_gram(XP[i])
        XP[i] = ' '.join(XP[i])
    with open('data/all_embeddings/all_proteins.txt' , 'w' , encoding='utf-8') as f:
        for i in XP:
            f.write(i + '\n')
    '''
    sentences = word2vec.Text8Corpus('data/all_embeddings/all_proteins.txt')
    model = word2vec.Word2Vec(sentences, size=30, hs=1, min_count=1, window=8 , iter=100)
    single_char = list(model.wv.vocab.keys())
    with open('data/all_embeddings/prots_chars.txt' , 'w') as f:
        for i in single_char:
            f.write(i + '\n')
    vector = np.zeros([len(single_char) , 30])
    for key in range(len(single_char)):
        vector[key] = model.wv[single_char[key]]
    np.savetxt('data/all_embeddings/prot_embedding.txt' , vector , fmt='%.8f' , delimiter='\t')
    model.save('data/all_embeddings/prot2vec.model')


def embedding_drug():
    '''
    XD1 = json.load(open("data/Davis/drugs.txt"))
    XD = list(XD1.values())
    XD2 = json.load(open("data/KIBA/drugs.txt"))
    XD.extend(list(XD2.values()))
    XD3 = json.load(open("data/DTINet/drugs.txt"))
    XD.extend(list(XD3.values()))
    for i in range(len(XD)):
        XD[i] = ' '.join(XD[i])
    with open('data/all_embeddings/all_drugs.txt' , 'w' , encoding='utf-8') as f:
        for i in XD:
            f.write(i + '\n')
    '''
    sentences = word2vec.Text8Corpus('data/all_embeddings/all_drugs.txt')
    model = word2vec.Word2Vec(sentences, size=30, hs=1, min_count=1, window=8 , iter=100)
    single_char = list(model.wv.vocab.keys())
    with open('data/all_embeddings/drugs_chars.txt' , 'w') as f:
        for i in single_char:
            f.write(i + '\n')
    vector = np.zeros([len(single_char) , 30])
    for key in range(len(single_char)):
        vector[key] = model.wv[single_char[key]]
    np.savetxt('data/all_embeddings/drugs_embedding.txt' , vector , fmt='%.8f' , delimiter='\t')
    model.save('data/all_embeddings/drug2vec.model')
    '''
    '''
def embedding():
    embedding_drug()
    embedding_protein()

if __name__ == "__main__":
    #create_pairs()
    embedding()

