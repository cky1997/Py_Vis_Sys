
#the Process of Lemmatisation
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# get the lexical properties of words
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# sentence = 'football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal.'
# sentence = 'studying'
sentence = 'government'
print(sentence)
tokens = word_tokenize(sentence)  # 分词
tagged_words = pos_tag(tokens)  # 获取单词的词性
print(tagged_words)

wnl = WordNetLemmatizer()
lemma_words = []

for tag in tagged_words:
    wordnet_pos = get_wordnet_pos(tag[1])
    lemma_words.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词性还原

print(lemma_words)

print("-"*20)
#the process of stemming
# Porter Stemmer基于Porter词干提取算法
lemma_words = ["government"]
porter_stemmer = PorterStemmer()
stemmed_words = []
for w in lemma_words:
    stemmed_words.append(porter_stemmer.stem(w))
print(stemmed_words)

