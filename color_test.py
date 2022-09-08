import os
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Mapping

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

#generate random colors
def random_color():
    return "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])

#读取数据
all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
all_data = all_data[pd.isna(all_data['title']) == False]
all_data = all_data[pd.isna(all_data['content']) == False]

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



# # 获得两篇文章的词频，返回类型Series
# text1_fre_series = pd.Series(selected_count_df[0]['frequency'].values, index=selected_count_df[0].iloc[:]["words"])
# text2_fre_series = pd.Series(selected_count_df[1]['frequency'].values, index=selected_count_df[1].iloc[:]["words"])
# print(text1_fre_series)
# print(text2_fre_series)

text1_fre_df = selected_count_df[0]
text2_fre_df = selected_count_df[1]
# print(text1_fre_df)
t1_fre_df_wd = pd.Series(text1_fre_df['frequency'].values, index=text1_fre_df.iloc[:]["words"])
t1_fre_df_wd = t1_fre_df_wd.sort_values(ascending=False)[:80]
# print(t1_fre_df_wd)
t2_fre_df_wd = pd.Series(text2_fre_df['frequency'].values, index=text2_fre_df.iloc[:]["words"])
t2_fre_df_wd = t2_fre_df_wd.sort_values(ascending=False)[:80]
# print("-"*30)
# print(text2_fre_df)
# print("-"*30)

inter_text_dict = reduce(lambda x, y: pd.merge(x, y, on=["words"], how='inner', suffixes=(None, '_x')),
                         selected_count_df)
# print(inter_text_dict.iloc[:, 1:])

inter_text_dict["frequency_total"] = inter_text_dict.iloc[:, 1:].sum(axis=1)
# print(inter_text_dict)

# 获得共同词， 返回list
inter_text_lst = np.array(inter_text_dict.iloc[:]["words"]).tolist()
print(inter_text_lst)


# 设置新列，确认是否为共同词，是 改为1
text1_is_in_common = [1 if text1_fre_df.iloc[i]["words"] in inter_text_lst else 0 for i in range(text1_fre_df.shape[0])]
text1_fre_df["is_in_common"] = pd.DataFrame(text1_is_in_common, columns = ['is_in_common'])
#text1_fre_df.groupby("is_in_common")["is_in_common"]类型为Series
text1_fre_df["color"] = text1_fre_df.groupby("is_in_common")["is_in_common"].\
    transform(lambda x: "#000000" if (x.index).all() else "#00FFFF")
# print(text1_fre_df)
# mapping color to words, type dict
t1_color_to_words = text1_fre_df.groupby("color")["words"].agg(list).to_dict()
print(t1_color_to_words)

text2_is_in_common = [1 if text2_fre_df.iloc[i]["words"] in inter_text_lst else 0 for i in range(text2_fre_df.shape[0])]
text2_fre_df["is_in_common"] = pd.DataFrame(text2_is_in_common, columns = ['is_in_common'])
text2_fre_df["color"] = text2_fre_df.groupby("is_in_common")["is_in_common"].\
    transform(lambda x: "#000000" if (x.index).all() else "#00FFFF")
# print(text2_fre_df)
t2_color_to_words = text2_fre_df.groupby("color")["words"].agg(list).to_dict()
print(t2_color_to_words)

print(text1_fre_df)
print(text2_fre_df)
# print(type(Counter(text1_fre_df.words)))


# wordcloud = WordCloud(width=500,
#                       height=500,
#                       max_words=80,
#                       min_word_length=3,
#                       prefer_horizontal=0.7,
#                       scale=30,
#                       background_color="rgba(255, 255, 255, 0)",
#                       mode="RGBA").generate_from_frequencies(t1_fre_df_wd)
# # mapping color to words, type dict
# # print(text1_fre_df.groupby("color")["words"].agg(list).to_dict())
#
# default_color = 'grey'
#
# # Create a color function with single tone
# customed_color_func = SimpleGroupedColorFunc(t1_color_to_words, default_color)
#
# # Apply our color function
# wordcloud.recolor(color_func=customed_color_func)
#
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# wordcloud.to_file(
#                     os.path.join("/Users/chenkanyu/Desktop/arti/archive/1.png"))
#
# ############
# wordcloud2 = WordCloud(width=500,
#                       height=500,
#                       max_words=80,
#                       min_word_length=3,
#                       prefer_horizontal=0.7,
#                       scale=30,
#                       background_color="rgba(255, 255, 255, 0)",
#                       mode="RGBA").generate_from_frequencies(t2_fre_df_wd)
# # mapping color to words, type dict
# # print(text1_fre_df.groupby("color")["words"].agg(list).to_dict())
#
# default_color = 'grey'
#
# # Create a color function with single tone
# customed_color_func = SimpleGroupedColorFunc(t2_color_to_words, default_color)
#
# # Apply our color function
# wordcloud2.recolor(color_func=customed_color_func)
#
# # plt.imshow(wordcloud, interpolation='bilinear')
# # plt.axis("off")
# # plt.show()
# wordcloud2.to_file(
#                     os.path.join("/Users/chenkanyu/Desktop/arti/archive/2.png"))
#




# inter_text_dict[]
# inter_text_dict[inter_text_dict['id'].isin(selected_id_list)].iloc[0]["content"]


# inter_text_dict.loc['frequency'] = inter_text_dict.iloc[:, 1:].sum(axis=1)
# inter_text_dict = inter_text_dict.iloc[:1, :]

# common_freq = inter_text_dict[["words", "frequency_total"]]
# common_freq_series = pd.Series(common_freq['frequency_total'].values, index=common_freq.iloc[:]["words"])
# print(common_freq_series)


#generate random colors
# def random_color():
#     print("#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]))

#random_color()








