# coding: utf-8

import pandas as pd
from bs4 import BeautifulSoup
import time

import warnings

warnings.filterwarnings("ignore")

start_time = time.time()

# 读取没有标签的数据   id	review
with open('../data/rawData/unlabeledTrainData.tsv', 'r', encoding='utf-8') as f:
    unlabeledTrain = [line.strip().split('\t') for line in f.readlines() if len(line.strip().split('\t')) == 2]

# 读取标记数据    id	sentiment	review
with open('../data/rawData/labeledTrainData.tsv', 'r', encoding='utf-8') as f:
    labeledTrain = [line.strip().split('\t') for line in f.readlines() if len(line.strip().split('\t')) == 3]

unlabel = pd.DataFrame(unlabeledTrain[1:], columns=unlabeledTrain[0])
label = pd.DataFrame(labeledTrain[1:], columns=labeledTrain[0])

# 数据预处理函数，处理一些特殊字符，进行简单的词形还原操作（这些可以应用nltk库帮助完成）
def cleanReview(subject):
    # 解析数据
    beau = BeautifulSoup(subject)

    new_Subject = beau.get_text()
    # 处理一些特殊符号
    new_Subject = new_Subject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')
    new_Subject = new_Subject.strip().split(" ")
    new_Subject = [word.lower() for word in new_Subject]
    new_Subject = " ".join(new_Subject)

    return new_Subject

unlabel["review"] = unlabel["review"].apply(cleanReview)
label["review"] = label["review"].apply(cleanReview)

# 将有标签的数据和无标签的数据进行合并
newDf = pd.concat([unlabel['review'], label['review']], axis=0)

# 保存为csv形式
newDf.to_csv('../data/processData/wordEmbedding.txt', index=False)

spend_time = time.time() - start_time

print("dataProcess time is ", spend_time)