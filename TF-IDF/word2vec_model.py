# -*- coding: utf-8 -*-
# 2022/5/12
# PyCharm
# author='Cao Jiawei',
# author_email='studyss@qq.com',
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 库名
# pip --default-timeout=100 install -U 库名
from gensim.models.word2vec import Word2Vec
import pandas as pd
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


# 构建word2vec模型，词向量的训练与生成
def get_dataset_vec(dataset):
    w2v_model = Word2Vec(dataset, vector_size=4,sg=1, min_count=2, hs=0)  # 初始化模型并训练
    w2v_model.save('word2vec_model.pkl')  # 保存训练结果

if __name__ == '__main__':
    # 数据集获取
    s = ["我来到北京清华大学", "他来到了网易杭研大厦", "小明硕士毕业与中国科学院","我爱北京天安门"] 
    corpusTest = Corpus(s)
    corpusTest.makeCorpus()
    sentences=corpusTest.getTokenizedCorpus()
    print(sentences)
    get_dataset_vec(sentences)
