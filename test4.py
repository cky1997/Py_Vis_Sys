import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans

vector=TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True)

train = ["Chinese Beijing Chinese",
         "Chinese Chinese Shanghai",
         "Chinese Macao",
         "Tokyo Japan Chinese"]

train2 = ["The flowers are beautiful.",
          "The name of these flowers is rose, they are very beautiful.",
          "Rose is beautiful.",
          "Are you like these flowers?"]


vector_fit = vector.fit_transform(train)
print(vector.get_feature_names())
print(vector_fit[0:1].toarray().tolist())
print("====------------------====")
#print(vector_fit.toarray().tolist())
print(vector_fit.toarray().tolist())

print("row:", np.size(vector_fit, 0))
print("col:", np.size(vector_fit, 1))
print("=========------------------------")


request = 'the bottle is'
request2 = 'rose beautiful flowers'
request3 = 'Chinese Beijing'
request_transform = vector.transform([request3])



print(request_transform.toarray().tolist())
print("row:", np.size(request_transform, 0))
print("col:", np.size(request_transform, 1))


tf_beijing = 1/3
idf_beijing = np.log(5/2)+1
tf_idf_beijing = tf_beijing*idf_beijing

tf_chinese = 2/3
idf_chinese = np.log(5/5)+1
tf_idf_chinese = tf_chinese*idf_chinese

norm = np.sqrt(tf_idf_beijing**2 + tf_idf_chinese**2)
norm = np.sqrt(idf_beijing**2 + idf_chinese**2)
print([idf_beijing, idf_chinese] / norm)
print([tf_idf_beijing, tf_idf_chinese] / norm)

print("----------------------------")
# tfidf_chinese = (1/3) * idf_chinese

# similarity = np.dot(request_transform,np.transpose(vector_fit))
# print(similarity)
# print("row:",np.size(similarity,0))
# print("col:",np.size(similarity,1))
#
#
#
# x = np.array(similarity.toarray()[0])
# print(x)
# indices=np.argsort(x)[-5:][::-1]
# print(indices)
#
# print("----------------------------")

# cosine_similarities = linear_kernel(vector_fit[0:1], vector_fit).flatten()
# print(cosine_similarities)
# related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != 0]
# print(related_docs_indices)
# print([index for index in related_docs_indices][0:5])

#print([index for index in related_docs_indices][0:5])
# print("----------------------------")











