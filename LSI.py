# -*- coding:utf-8 -*-
import nltk
file = open("coursera_corpus.txt")
courses = [line.strip() for line in file]
courses_name = [course.split('\t')[0] for course in courses]
print(courses_name[0:10])
text_lower = [ [ word for word in document.lower().split()] for document in courses]
from nltk.tokenize import word_tokenize
texts_tokenized = [[word.lower() for word in word_tokenize(document)]for document in courses]
print(texts_tokenized)
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords]for document in texts_tokenized]
print(texts_filtered_stopwords)
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
texts_filtered = [[word for word in document if not word in english_punctuations]for document in texts_filtered_stopwords]
print(texts_filtered)
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
text_stemmed = [[st.stem(word)for word in document]for document in texts_filtered]
all_stems = sum(text_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stems_once] for text in text_stemmed]
#print(texts)
from gensim import corpora,models, similarities
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text)for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf =tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=10)
index = similarities.MatrixSimilarity(lsi[corpus])
print(courses_name[210])
ml_course = texts[189]
ml_bow = dictionary.doc2bow(ml_course)
ml_lsi = lsi[ml_bow]
print(ml_lsi)
sims = index[ml_lsi]
sort_sims = sorted(enumerate(sims),key = lambda  item: -item[1])
print(sort_sims[0:10])
print(courses_name[210])
print(courses_name[174])
print(courses_name[63])
print(courses_name[141])
print(courses_name[184])