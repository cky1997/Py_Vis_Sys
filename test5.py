from functools import reduce

import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

# X = pd.read_csv("/Users/chenkanyu/Desktop/arti/archive/articles1.csv")
# X = X[pd.isna(X['title'])==False]
# X = X[pd.isna(X['content'])==False]
# #
# # print(X.iloc[34339])
#
# text_content = X['content']
# vector=TfidfVectorizer(use_idf=True, smooth_idf=True, norm="l2", lowercase=True)
# tfidf = vector.fit_transform(text_content)
#
# # Request function : search the top_n articles from a request ( request = string)
# def search(tfidf_matrix,model,request, top_n = 5):
#     request_transform = model.transform([request])
#     #获得相似度
#     similarity = np.dot(request_transform, np.transpose(tfidf_matrix))
#     # print("row:", np.size(similarity, 0))
#     # print("col:", np.size(similarity, 1))
#     # print(similarity.toarray().shape)
#     x = np.array(similarity.toarray()[0])
#
#     #print(x)
#     print(np.argsort(x)[-5:])
#     indices=np.argsort(x)[-5:][::-1]
#     print(type(indices))
#     return indices
#
# Find similar : get the top_n articles similar to an article
# def find_similar(tfidf_matrix, index, top_n = 5):
#     cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
#     # print(cosine_similarities)
#     related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
#     # print(related_docs_indices)
#     # return [index for index in related_docs_indices][0:top_n]
#     return related_docs_indices[0:top_n]
#
# # Print the result
# def print_result(request_content,indices,X):
#     print('\nsearch : ' + request_content)
#     print('\nBest Results :')
#     for i in indices:
#         print('id = {0:5d} - title = {1}'.format(i,X['title'].loc[i]))
#
#
# index = 13084
# result = find_similar(tfidf, index, top_n = 5)
# print_result('13084 - title = Accusations of Anti-Semitism Taint French Presidential Race',result,X)
# request = 'peillon macron fillon marche'
#
# result = search(tfidf,vector, request, top_n = 5)
# print_result(request,result,X)

# import numpy as np
# a=np.random.randint(15,size=(1,10))
# print(a)
# print(a[0])
# print(np.argsort(a)[-5:])

# import numpy as np
# import pandas as pd
all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
all_data = all_data[pd.isna(all_data['title']) == False]
all_data = all_data[pd.isna(all_data['content']) == False]
# #print(all_data.iloc[0:6, 1:3])
#
# index = all_data[all_data["id"] == 31821].index.tolist()[0]
# print(index)
# print(type(index))



# total_id = []
# for i in range(all_data.shape[0]):
#     total_id.append(all_data.iloc[i]["id"])
# #total_id = all_data["id"]
# if 73688 in total_id:
#     print("Yes")
# else:
#     print("No")
# print(total_id)
# print(len(total_id))
# print(type(total_id))

#print(all_data.iloc[0]["id"])


#
# all_text_content = all_data['content']
#
# index = ["17283", "17285", "17286", "17288"]
#
# #print(type(all_data))
# searched_data = all_data
# # searched_data = all_data.iloc[index, 1:3]
# searched_data = all_data[all_data['id'].isin([17283])]
# #a = all_data[all_data['id'].isin([17283])].loc[0].loc["content"]
# b = all_data[all_data['id'].isin([int("17283")])].iloc[0]["content"]
# # searched_data = all_data[0:1, 1:3]
# #print(all_data.head(10))
# print(b)
# # print(a)
# # print(type(all_data.loc[0].loc["content"]))
# # print(len(index))



# index = "13084"
# print(type(int(index)))


# self.all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
# self.all_data = self.all_data[pd.isna(self.all_data['title']) == False]
# self.all_data = self.all_data[pd.isna(self.all_data['content']) == False]

# selected_row_id = self.tableWidget.selectedItems()[0].text()
# print(selected_row_id)
# print(type(selected_row_id))

# text_temp = all_data[all_data['id'].isin([int(selected_row_id)])].iloc[0]["content"]
# text_temp = all_data.iloc[0]["content"]

# 获得被选中的ID List
# selected_id =

#这里用索引测试
# selected_id = [53775, 53892, 55152, 66815, 54940]
selected_id_lst = [17283]
coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
# text = []
selected_count_df = []
for i in range(len(selected_id_lst)):
    text = []
    # 某一篇的文章内容
    text_content = all_data[all_data['id'].isin(selected_id_lst)].iloc[i]["content"]
    text.append(text_content)
    count_matrix = coun_vect.fit_transform(text)
    count_array = count_matrix.toarray()
    #selected_count_array.append(count_array)
    df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
    text_dict = df.iloc[0].sort_values(ascending=False)[:80]
    # data = count_array, columns = coun_vect.get_feature_names_out()
    text_dict_df = pd.DataFrame({'words': text_dict.index, 'frequency': text_dict.values})
    selected_count_df.append(text_dict_df)

    # print(type(count_array))
    # print(text_dict_df)
    # print("-"*20)

# print(selected_count_df)
inter_text_dict = reduce(lambda x, y: pd.merge(x, y, on=["words"], how='inner', suffixes=(None, '_x')), selected_count_df)
# print(inter_text_dict.iloc[:, 1:])
inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:].sum(axis=1)
# inter_text_dict.loc['frequency'] = inter_text_dict.iloc[:, 1:].sum(axis=1)
# inter_text_dict = inter_text_dict.iloc[:1, :]
common_freq = inter_text_dict[["words", "frequency_total"]]
# print(inter_text_dict.iloc[:, :2])
print(common_freq)
# print("-"*10)
# print(common_freq.iloc[:]["words"])
# print(type(common_freq.iloc[:]["words"]))
# print("-"*10)
common_freq_series = pd.Series(common_freq['frequency_total'].values, index=common_freq.iloc[:]["words"])

common_text_dict = common_freq_series.sort_values(ascending=False)[:80]
print(common_text_dict)
common_text_dict = common_text_dict[common_text_dict > 3]

wordcloud = WordCloud(width=500,
                              height=500,
                              max_words=50,
                              min_word_length=3,
                              prefer_horizontal=0.7,
                              scale=30,
                              background_color="rgba(255, 255, 255, 0)",
                              mode="RGBA").generate_from_frequencies(common_text_dict)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# a = pd.DataFrame(data=common_freq["frequency_total"], columns=common_freq["words"])
# print(common_freq_series[["macron"]])
# print(common_freq["words"].T)
# common_freq = [common_freq['frequency_total'].values]
# print(common_freq.to_numpy())

# common_freq = pd.DataFrame(data=common_freq['frequency_total'].values, columns=common_freq["words"])
# print(common_freq)

# print(common_freq_series)
# print(common_freq['words'])

# print(type(common_freq))common_freq['frequency_total'].values

# df = pd.DataFrame(data=selected_count_array[0], columns=coun_vect.get_feature_names_out())
# print(df.iloc[0])
# print(df)
#text_dict = df.iloc[0].sort_values(ascending=False)[:50]








#
# count_matrix = []
# for i in text:
#     count_matrix.append(coun_vect.fit_transform(i))
#
#
#
# df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
# # print(df.iloc[0])
#
# text_dict = df.iloc[0].sort_values(ascending=False)[:50]
# # text_temp = all_data[all_data['id'].isin(selected_id)].iloc[0]["content"]
# # text_temp1 = all_data[all_data['id'].isin(selected_id)].iloc[1]["content"]
# # text_temp2 = all_data[all_data['id'].isin(selected_id)].iloc[2]["content"]
#
# # text_temp = all_data[all_data['id'].isin([17283, 17824, 17825])]["content"]
# print(text)







# print(selected_row_id)
# print(text_temp)

# text = []
# text.append(text_temp)
#
#
# coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
# count_matrix = coun_vect.fit_transform(text)
# count_array = count_matrix.toarray()
# df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
# # print(df.iloc[0])
#
# text_dict = df.iloc[0].sort_values(ascending=False)[:50]
# text_dict = text_dict[text_dict > 3]




