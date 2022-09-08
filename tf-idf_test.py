from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


data = ["Chinese Beijing Chinese",
         "Chinese Chinese Shanghai",
         "Chinese Macao",
         "Tokyo Japan Chinese"]
vector=TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True)
vector_fit = vector.fit_transform(data)

print(vector.get_feature_names())
print(vector_fit[0:1].toarray().tolist())

tf_beijing = 1/3
idf_beijing = np.log(5/2)+1
tf_idf_beijing = tf_beijing*idf_beijing

tf_chinese = 2/3
idf_chinese = np.log(5/5)+1
tf_idf_chinese = tf_chinese*idf_chinese

# norm = np.sqrt(tf_idf_beijing**2 + tf_idf_chinese**2)
# norm = np.sqrt(idf_beijing**2 + idf_chinese**2)
# print([idf_beijing, idf_chinese] / norm)
norm = np.sqrt(tf_idf_beijing**2 + tf_idf_chinese**2)
print([tf_idf_beijing, tf_idf_chinese] / norm)

