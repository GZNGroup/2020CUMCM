import gensim
import logging
import numpy as np

model = gensim.models.Word2Vec.load("C:/Users/guozn/Desktop/wiki_model")

catag = {}
catag["商业"] = ["商", "贸易", "销售", "经营"]
catag["服务业"] = ["物流", "房地产", "酒店", "服务", "事务", "招标"]
catag["工程"] = ["路", "桥", "建", "环境", "装", "林"]
catag["医药"] = ["医", "药", "卫生"]
catag["技术"] = ["科技", "电子", "通信", "网络"]
catag["文娱"] = ["文", "娱乐", "影视", "广告", "演艺"]


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.0
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.0
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


names = [u"商业", u"服务业", u"工程", u"医药", u"技术", u"文娱"]
center = []
for key, value in catag.items():
    for i in value:
        center.append(model[i])

from sklearn.decomposition import PCA
from matplotlib import pyplot
from pandas import DataFrame

# 基于2d PCA拟合数据
pca = PCA(n_components=2)
X = center
print(X)
result = pca.fit_transform(X)
print(result, names)
# 可视化展示
pyplot.scatter(result[:, 0], result[:, 1])
words = list(X)
for i, word in enumerate(words):
    pyplot.annotate(names[i], xy=(result[i, 0], result[i, 1]))
pyplot.show()
