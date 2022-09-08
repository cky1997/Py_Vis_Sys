
#对dataframe的测试

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import os
import matplotlib.pyplot as plt

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
X = pd.read_csv("/Users/chenkanyu/Desktop/arti/archive/articles1.csv")
X = X[pd.isna(X['title'])==False]
X = X[pd.isna(X['content'])==False]
#获得第一行的id
print(str(X.loc[0].loc["id"]))
print(type(str(X.loc[0].loc["id"])))
print(type("asd"))
# #获得第一行的content
# print(X.loc[0].loc["content"])
#
#
text_temp = X.loc[1].loc["content"]
text = []
text.append(text_temp)

coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
#print(df.iloc[0])


text_dict = df.iloc[0].sort_values(ascending=False)[:50]
text_dict = text_dict[text_dict>3]
#print(text_dict)
wordcloud = WordCloud(width=500,
                      height=500,
                      max_words=50,
                      min_word_length=3,
                      scale=25,
                      background_color="rgba(255, 255, 255, 0)",
                      mode="RGBA").generate_from_frequencies(text_dict)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file(os.path.join("/Users/chenkanyu/Desktop/arti/archive/"+str(X.loc[1].loc["id"])+".png"))






