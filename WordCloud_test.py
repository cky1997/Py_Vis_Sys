from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

#txt文件测试
text_temp = open("/Users/chenkanyu/Desktop/arti/archive/test.txt").read()
text = []
text.append(text_temp)

#
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
X = pd.read_csv("Desktop/arti/archive/articles1.csv")
X = X[pd.isna(X['title'])==False]
X = X[pd.isna(X['content'])==False]
print(X.loc[0])

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
wordcloud.to_file("/Users/chenkanyu/Desktop/arti/archive/test.png")

#print(STOPWORDS)







