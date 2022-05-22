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
    n_dim = 300
    w2v_model = Word2Vec(dataset, vector_size=4,sg=1, min_count=2, hs=0)  # 初始化模型并训练
    # 在测试集上训练
    # w2v_model.train(x_test,total_examples=w2v_model.corpus_count,epochs=w2v_model.iter) #追加训练模型
    # 将imdb_w2v模型保存，训练集向量，测试集向量保存到文件
    # print(w2v_model['会议'])
    w2v_model.save('word2vec_model.pkl')  # 保存训练结果

def datlist_dic(datlist):
    train_data =[]
    dataset=[]
    for dat in datlist:
        print(dat)
        train_data['word']=train_data['contents'].apply(dat)
        dataset = pd.concat([dataset, train_data])
    return train_data
if __name__ == '__main__':
    # 数据集获取
    s = ["我来到北京清华大学", "他来到了网易杭研大厦", "小明硕士毕业与中国科学院","我爱北京天安门"] 
    # raw_sentences = ['the quick brown fox jump over the lazy dogs', 'yoyoyo you go home now to sleep']  # 输入两句话
    # sentences = [s.split() for s in raw_sentences]  # 分词
    # print(sentences)
    corpusTest = Corpus(s)
    corpusTest.makeCorpus()
    sentences=corpusTest.getTokenizedCorpus()
    print(sentences)
    # print(datlist_dic(datlist))
    # word2vec词向量训练
    get_dataset_vec(sentences)

