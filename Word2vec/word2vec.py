# -*- coding: utf-8 -*-
# 2022/5/12
# PyCharm
# author='Cao Jiawei',
# author_email='studyss@qq.com',
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 库名
# pip --default-timeout=100 install -U 库名
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import jieba
class Corpus(object):
    def __init__(self, texts):
        self.texts = texts
        self.tokenizedCorpus = []

    def makeCorpus(self):
        for sentence in self.texts:
            self.tokenizedCorpus.append([])
            for x in jieba.tokenize(sentence):
                if x[0] == ' 'or x[0] == '/' or x[0] == '）' or x[0] == '（' or x[0] == '，' or x[0] == ')' or x[0] == '(':
                    # 我们句子中的"Python"和"Numpy"之间有空格，而jieba认为这个空格也是一个词，但我们不需要空格作为单独的词
                    continue
                self.tokenizedCorpus[-1].append(x[0])

    def getTokenizedCorpus(self):
        return self.tokenizedCorpus

# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(sentence, size, w2v_model):
    sen_vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in sentence:
        try:
            sen_vec += w2v_model.wv[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec


# 将文本数据转换为文本向量
def doc_vec():
    dat = ["我来到北京清华大学", "他来到了网易杭研大厦", "小明硕士毕业与中国科学院","我爱北京天安门"] 
    w2v_model = Word2Vec.load('word2vec_model.pkl')  # 加载训练好的Word2Vec模型

    # 读取词权重字典
    # with open('data/key_words_importance', 'r') as f:
    # key_words_importance = eval(f.read())
    corpusTest = Corpus(dat)
    corpusTest.makeCorpus()
    train_data_list = corpusTest.getTokenizedCorpus()
    # 训练集转换为向量
    train_docvec_list = np.concatenate([build_sentence_vector(sen, 4, w2v_model) for sen in train_data_list])

    return train_docvec_list

if __name__ == "__main__":
    Res=doc_vec()
    np.save('average.npy', Res)
    x = np.load('average.npy')
    print(x)