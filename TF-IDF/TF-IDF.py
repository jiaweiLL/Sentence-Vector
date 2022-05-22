# -*- coding: utf-8 -*-
# 2022/5/12
# PyCharm
# author='Cao Jiawei',
# author_email='studyss@qq.com',
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 库名
# pip --default-timeout=100 install -U 库名
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.word2vec import Word2Vec
import pandas as pd
import math
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


# 对每个句子的所有词向量取加权均值，来生成一个句子的vector
def build_sentence_vector_weight(sentence, word, size, w2v_model, key_weight):
    sen_vec = np.zeros(size).reshape((1, size))
    count = 0
    for sen in range(len(sentence)):
        try:
            index=0
            for i in range(len(word)):
                if word[i]==sentence[sen]:
                    index=i
                    break
            if key_weight[index]!=0:
                sen_vec += (np.dot(w2v_model.wv[sentence[sen]], math.exp(key_weight[index]))).reshape((1, size))
                count += 1
            else:
                sen_vec += w2v_model.wv[sentence[sen]].reshape((1, size))
                count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec

# 将文本数据转换为文本向量
def doc_vec():
    dat = ["我来到北京清华大学", "他来到了网易杭研大厦", "小明硕士毕业与中国科学院","我爱北京天安门"] 
    corpusTest = Corpus(dat)
    corpusTest.makeCorpus()
    datlist=corpusTest.getTokenizedCorpus()
    dataset = [str]* len(dat)
    for index in range(len(dat)):
        dataset[index]=''
        for data in datlist[index]:
            dataset[index]=dataset[index]+' '+data
        dataset[index]=dataset[index].strip()
    w2v_model = Word2Vec.load('word2vec_model.pkl')  # 加载训练好的Word2Vec模型
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(dataset))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])

    corpusTest = Corpus(dat)
    corpusTest.makeCorpus()
    train_data_list = corpusTest.getTokenizedCorpus()
    # 训练集转换为向量
    train_docvec_list = np.concatenate([build_sentence_vector_weight(train_data_list[sen], word, 4, w2v_model,weight[sen]) for sen in range(len(train_data_list))])

    return train_docvec_list

if __name__ == "__main__":
    Res=doc_vec()
    np.save('IF-IDF.npy', Res)
    x = np.load('IF-IDF.npy')
    print(x)