import numpy as np
import string
import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('stopwords')
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.metrics.pairwise import linear_kernel

# https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html
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


class SimilarityTools:

    # Request function : search the top 5 articles from a request
    @staticmethod
    def search(tfidf_matrix, vectorizer, request):
        request_transform = vectorizer.transform([request])  # calculate idf
        similarity = np.dot(request_transform, np.transpose(tfidf_matrix))
        similarity_array = np.array(similarity.toarray()[0])
        indices = np.argsort(similarity_array)[::-1][:5]
        return indices

    # Find similar : search the top 5 articles similar to an article
    @staticmethod
    def find_similar(tfidf_matrix, index):
        cosine_similarities = linear_kernel(tfidf_matrix[index:index + 1], tfidf_matrix).flatten()
        related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
        return related_docs_indices[0:5]

    # # Print the result
    # @staticmethod
    # def print_result(request_content,indices,dataset):
    #     print('\nRequest : ' + request_content)
    #     print('\nResults :')
    #     for i in indices:
    #         print('index = {0} - title = {1}'.format(dataset['id'].loc[i], dataset['title'].loc[i]))


nlp = spacy.load('en_core_web_md')


class Preprocessing:

    @staticmethod
    def lemm(text):
        doc = nlp(text)
        # for word in doc:
        #     print(word.text, word.pos_)
        # print(" ".join([token.lemma_ for token in doc]))
        return " ".join([token.lemma_ for token in doc])

    @staticmethod
    def trans_Ch_to_Eng(text):
        table = {ord(i): ord(j) for i, j in zip(
            u'’，。！？【】（）％＃＠＆１２３４５６７８９０“”—',
            u'\',.!?[]()%#@&1234567890""-')}
        text_dic = text.maketrans(table)
        return text.translate(text_dic)

    stop_words = stopwords.words("english")

    @staticmethod
    def remove_stopwords(text, stop_words=stop_words):
        text = ' '.join([word for word in text.split() if word not in stop_words])
        # print(text)
        return text

    @staticmethod
    def remove_punct(df):
        # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        punct = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
        transtab = str.maketrans(dict.fromkeys(punct, ''))

        df['content'] = '【】'.join(df['content'].tolist()).translate(transtab).split('【】')

    # MaxU's version (https://stackoverflow.com/a/50444659/4909087)
    @staticmethod
    def remove_apostro(df):
        punct = string.punctuation
        transtab = str.maketrans(dict.fromkeys(punct, ''))
        return df.assign(content=df['content'].str.translate(transtab))

    @staticmethod
    def remove_single_char(text):
        text = ' '.join([word for word in text.split() if len(word) > 1])
        return text


# def remove_single_chr(df):
#     punct = string.ascii_uppercase+string.ascii_lowercase
#     transtab = str.maketrans(dict.fromkeys(punct, ''))
#     return df.assign(content=df['content'].str.translate(transtab))

# all_data = pd.read_csv('/Users/chenkanyu/Desktop/arti/archive/articles1.csv')
# all_data = all_data[pd.isna(all_data['title']) == False]
# all_data = all_data[pd.isna(all_data['content']) == False]
#
# # Lowercase
# all_data['content'] = all_data['content'].apply(np.char.lower)
#
# # Chinese to English
# all_data = all_data.assign(content=all_data['content'].apply(Preprocessing.trans_Ch_to_Eng))
# print(all_data.loc[1]['content'])
#
# # Remove stopwords
# all_data = all_data.assign(content=all_data['content'].apply(Preprocessing.remove_stopwords))
# print(all_data.loc[1]['content'])
#
# # Remove punctuation
# Preprocessing.remove_punct(all_data)
# print(all_data.loc[1]['content'])
#
# # Remove stopwords
# all_data = all_data.assign(content=all_data['content'].apply(Preprocessing.remove_stopwords))
# print(all_data.loc[1]['content'])
#
# # Lemm
# all_data = all_data.assign(content=all_data['content'].apply(Preprocessing.lemm))
# print(all_data.loc[1]['content'])
#
# # Remove apostrophe
# all_data = Preprocessing.remove_apostro(all_data)
# print(all_data.loc[1]['content'])
#
# # Remove single character
# all_data = all_data.assign(content=all_data['content'].apply(Preprocessing.remove_single_char))
# print(all_data.loc[1]['content'])
#
# all_data.iloc[:, 1:].to_csv("processed_dataset_new.csv")
