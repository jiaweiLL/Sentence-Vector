# -*- coding: utf-8 -*-
# 2022/4/18
# PyCharm
# author='Cao Jiawei',
# author_email='studyss@qq.com',
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 库名
# pip --default-timeout=100 install -U 库名
import jieba
import gensim
from  gensim.models.doc2vec import Doc2Vec
import numpy as np
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
s=["我来到北京清华大学", "他来到了网易杭研大厦", "小明硕士毕业与中国科学院","我爱北京天安门"] 
corpusTest = Corpus(s)
corpusTest.makeCorpus()
print(corpusTest.getTokenizedCorpus())
TaggededDocument=gensim.models.doc2vec.TaggedDocument
def X_train(cut_sentence):
    x_train=[]
    for i,text in enumerate(cut_sentence):
        word_list=text
        l=len(text)
        word_list[l-1]=word_list[l-1].strip()
        document=TaggededDocument(word_list,tags=[i])
        x_train.append(document)
    return x_train
C=X_train(corpusTest.getTokenizedCorpus())
def trian(x_train,size=4):
    model=Doc2Vec(x_train,min_count=1,vector_size=size,window=3,sample=1e-3,hs=1,workers=4)
    model.train(x_train,total_examples=model.corpus_count,epochs=20)
    return model
model_dm=trian(C)

def getRes():
    list=[]
    test_text=corpusTest.getTokenizedCorpus()
    for text in test_text:
        inferred_vector_dm = model_dm.infer_vector(text)  ##得到文本的向量
        list.append(inferred_vector_dm.tolist())
    return list
Res=getRes()
np.save('dat.npy', Res)
x=np.load('dat.npy')
print(x)
# for res in Res:
#     print(res)
# print(len(Res))
