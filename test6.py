from functools import reduce
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

# #单篇文章的词频
#
from wordcloud import WordCloud

all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
all_data = all_data[pd.isna(all_data['title']) == False]
all_data = all_data[pd.isna(all_data['content']) == False]

# text_temp = all_data.iloc[0]["content"]
# # print(selected_row_id)
# # print(text_temp)
#
# text = []
# text.append(text_temp)
#
#
# coun_vect = CountVectorizer(stop_words="english", ngram_range=(1, 3))
# count_matrix = coun_vect.fit_transform(text)
# # print(count_matrix)
# count_array = count_matrix.toarray()
# # print(count_array)
# # print(type(count_array))
# df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
# # # print(df.iloc[0])
# # print(type(coun_vect.get_feature_names_out()))
# print(df)
# # print(type(df))
# print(df.iloc[0])
# # print(type(df.iloc[0]))
# #
# # text_dict = df.iloc[0].sort_values(ascending=False)[:50]
# # print(type(text_dict))
# # print(text_dict)
#
#
#
# # a=[1,2,3,4]
# # b=[1,5,6]
# # print(list(set(a)&set(b)))


# a = [1, 2, 3, 4, 5]
# print("the value is "+str(a[0]))

#词云中词的颜色调试
# selected_id_lst = [53775, 53892, 55152, 66815, 54940]
selected_id_lst = [53775, 53892]

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
    # selected_count_array.append(count_array)
    df = pd.DataFrame(data=count_array, columns=coun_vect.get_feature_names_out())
    text_dict = df.iloc[0].sort_values(ascending=False)[:80]
    # data = count_array, columns = coun_vect.get_feature_names_out()
    text_dict_df = pd.DataFrame({'words': text_dict.index, 'frequency': text_dict.values})
    selected_count_df.append(text_dict_df)
    # 这里可以考虑
    # selected_count_df.append(text_dict_df[text_dict_df['frequency']>3])

    # print(type(count_array))
    # print(text_dict_df)
    # print("-"*20)

# print(selected_count_df)
# print(selected_count_df[0])
# print(selected_count_df[1])


# 获得两篇文章的词频，返回类型Series
text1_fre_series = pd.Series(selected_count_df[0]['frequency'].values, index=selected_count_df[0].iloc[:]["words"])
text2_fre_series = pd.Series(selected_count_df[1]['frequency'].values, index=selected_count_df[1].iloc[:]["words"])
print(text1_fre_series)
print(text2_fre_series)
print("-"*30)
inter_text_dict = reduce(lambda x, y: pd.merge(x, y, on=["words"], how='inner', suffixes=(None, '_x')),
                         selected_count_df)
# print(inter_text_dict.iloc[:, 1:])
inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:].sum(axis=1)
# inter_text_dict.loc['frequency'] = inter_text_dict.iloc[:, 1:].sum(axis=1)
# inter_text_dict = inter_text_dict.iloc[:1, :]

common_freq = inter_text_dict[["words", "frequency_total"]]
common_freq_series = pd.Series(common_freq['frequency_total'].values, index=common_freq.iloc[:]["words"])
print(common_freq_series)

color_list = ['#191970', '#00FFFF']#建立颜色数组
colormap = mcolors.ListedColormap(color_list)#调用
wordcloud1 = WordCloud(width=500,
                      height=500,
                      max_words=50,
                      min_word_length=3,
                      prefer_horizontal=0.7,
                      scale=30,
                      colormap=colormap,#设置颜色
                      background_color="rgba(255, 255, 255, 0)",
                      mode="RGBA").generate_from_frequencies(text1_fre_series)
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud1.to_file(
                    os.path.join("/Users/chenkanyu/Desktop/arti/archive/1.png"))


# wordcloud2 = WordCloud(width=500,
#                       height=500,
#                       max_words=50,
#                       min_word_length=3,
#                       prefer_horizontal=0.7,
#                       scale=30,
#                       colormap=colormap,#设置颜色
#                       background_color="rgba(255, 255, 255, 0)",
#                       mode="RGBA").generate_from_frequencies(text2_fre_series)
# plt.imshow(wordcloud2, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# # print("-"*10)
# # print(common_freq.iloc[:]["words"])
# # print(type(common_freq.iloc[:]["words"]))
# # print("-"*10)
# common_freq_series = pd.Series(common_freq['frequency_total'].values, index=common_freq.iloc[:]["words"])
#
# common_text_dict = common_freq_series.sort_values(ascending=False)[:80]
# print("-"*30)
# print(common_text_dict)
# # common_text_dict = common_text_dict[common_text_dict > 3]
# #
# # wordcloud = WordCloud(width=500,
# #                       height=500,
# #                       max_words=50,
# #                       min_word_length=3,
# #                       prefer_horizontal=0.7,
# #                       scale=30,
# #                       background_color="rgba(255, 255, 255, 0)",
# #                       mode="RGBA").generate_from_frequencies(common_text_dict)
# # plt.imshow(wordcloud, interpolation='bilinear')
# # plt.axis("off")
# # plt.show()


