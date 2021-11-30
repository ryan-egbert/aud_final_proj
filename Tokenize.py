import nltk
import pandas as pd
import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Create a dataframe from csv
df = pd.read_csv('data_csv.csv', delimiter=',', names=["Category", "Desc", "ID"])
job_list = df[df["Category"] != "resumes / job wanted"]
resume_list = df[df["Category"] == "resumes / job wanted"]

#Part 1. Text representation (total 5 points)
#1. Tokenize each post in the collection.
#2. Use the tokenized reviews after step 1, lemmatize all the words, convert in lowercase.
#3. Based on the output in step 2, remove all the stop-words and the punctuations.

processed_collection_r = []
processed_collection_j = []
lemmatizer = nltk.stem.WordNetLemmatizer()

for post in resume_list.Desc:
    #tokenize each review
    tokens = nltk.word_tokenize(post)

    #lemmatize each review and convert it to lower
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    #Remove Stop Words and Punctuations
    tokens = [token for token in tokens if not token in stopwords.words('english') if token.isalpha()]

    joins = " ".join(tokens)
    processed_collection_r.append(joins)

for post in job_list.Desc:
    #tokenize each review
    tokens = nltk.word_tokenize(post)

    #lemmatize each review and convert it to lower
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    #Remove Stop Words and Punctuations
    tokens = [token for token in tokens if not token in stopwords.words('english') if token.isalpha()]

    joins = " ".join(tokens)
    processed_collection_j.append(joins)

#print(processed_collection)

#4. Based on the output in step 3, convert each of the reviews in a TD-IDF vector.
#The minimal document frequency for each term is 3. Also, include 2-gram.
#Tfidf
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
vectorizer.fit(processed_collection_r)
v_r = vectorizer.transform(processed_collection_r)
print("[", len(v_r.toarray()),",", len(vectorizer.vocabulary_),"]")

with open('tfidf_output_r.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(v_r.toarray())

vectorizer2 = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
vectorizer2.fit(processed_collection_j)
v_j = vectorizer2.transform(processed_collection_j)
print("[", len(v_j.toarray()),",", len(vectorizer2.vocabulary_),"]")

with open('tfidf_output_j.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(v_j.toarray())

#5. Based on the output in step 1, POS-tag each word and do a TD-IDF vectorization,
# the minimal document frequency for each term is 4 (please don’t do normalization and stopword removal)
#POS TAG
POS_Collection_r = []
for post in resume_list.Desc:
    #tokenize each review
    token_doc = nltk.word_tokenize(post)
    POS_token_doc = nltk.pos_tag(token_doc)
    #print(POS_token_doc)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS_Collection_r.append(" ".join(POS_token_temp))

#print(POS_Collection)
vectorizer3 = TfidfVectorizer(min_df=1)
vectorizer3.fit(POS_Collection_r)
POS_v_r = vectorizer3.transform(POS_Collection_r)
print("[", len(POS_v_r.toarray()),",", len(vectorizer3.vocabulary_),"]")

with open('POS_TAG_output_r.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(POS_v_r.toarray())

POS_Collection_j = []
for post in job_list.Desc:
    #tokenize each review
    token_doc = nltk.word_tokenize(post)
    POS_token_doc = nltk.pos_tag(token_doc)
    #print(POS_token_doc)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS_Collection_j.append(" ".join(POS_token_temp))

#print(POS_Collection)
vectorizer4 = TfidfVectorizer(min_df=1)
vectorizer4.fit(POS_Collection_j)
POS_v_j = vectorizer4.transform(POS_Collection_j)
print("[", len(POS_v_j.toarray()),",", len(vectorizer4.vocabulary_),"]")

with open('POS_TAG_output_j.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(POS_v_j.toarray())

## PART 2

# use index-based encoding to encode each word and represent each text document in a vector(list) of indices (in integer),
# save the representation of the whole collection as a 2D array (i.e., a matrix).
# Index Encoding 2-D
tokenized_r = [nltk.word_tokenize(post) for post in resume_list.Desc]
# A set for all possible words
words = [j for i in tokenized_r for j in i]
# define vocabulary
index_encoder = LabelEncoder()
index_encoder = index_encoder.fit(words)
# encoding and trasforming from array to list
index_encoded = [index_encoder.transform(doc).tolist() for doc in tokenized_r]

# padding
max = len(index_encoded[0])
for r in index_encoded:
    if len(r)>max:
        max = len(r)
for i in range(0,len(index_encoded)):
    if len(index_encoded[i]) < max:
        zeros = ((max-len(index_encoded[i])) * [0])
        index_encoded[i] = [*zeros, *index_encoded[i]]

with open('Index_Encoding_output_r.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(index_encoded)
print("[", len(index_encoded),",", len(index_encoded[0]),"]")

#Index Encoding for Jobs
tokenized_j = [nltk.word_tokenize(post) for post in job_list.Desc]
# A set for all possible words
words = [j for i in tokenized_j for j in i]
# define vocabulary
index_encoder_j = LabelEncoder()
index_encoder_j = index_encoder_j.fit(words)
# encoding and trasforming from array to list
index_encoded_j = [index_encoder_j.transform(doc).tolist() for doc in tokenized_j]

# padding
max = len(index_encoded_j[0])
for r in index_encoded_j:
    if len(r)>max:
        max = len(r)
for i in range(0,len(index_encoded_j)):
    if len(index_encoded_j[i]) < max:
        zeros = ((max-len(index_encoded_j[i])) * [0])
        index_encoded_j[i] = [*zeros, *index_encoded_j[i]]

with open('Index_Encoding_output_j.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(index_encoded_j)
print("[", len(index_encoded_j),",", len(index_encoded_j[0]),"]")

# Based on the output of step 1(in Part 2), use one-hot encoding for each index to further
# represent each text document as a one-hot 2D array, save the representation of the whole
# collection as a 3D array (i.e., a cube).

#One-Hot encoding
# vocabulary
indices_list = [[j] for i in index_encoded for j in i]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder = onehot_encoder.fit(indices_list)
# encoding
onehot_encoded1 = [onehot_encoder.transform([[i] for i in doc_i]).tolist() for doc_i in index_encoded]

with open('One_Hot_Encoding_output_r.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(onehot_encoded1)

print("[", len(onehot_encoded1),",", len(onehot_encoded1[0]),",", len(onehot_encoded1[0][0]),"]")

#One-Hot encoding for jobs
# vocabulary
indices_list_j = [[j] for i in index_encoded_j for j in i]
onehot_encoder_j = OneHotEncoder(sparse=False)
onehot_encoder_j = onehot_encoder_j.fit(indices_list_j)
# encoding
onehot_encoded1_j = [onehot_encoder_j.transform([[i] for i in doc_i]).tolist() for doc_i in index_encoded_j]

with open('One_Hot_Encoding_output_j.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(onehot_encoded1_j)

print("[", len(onehot_encoded1_j),",", len(onehot_encoded1_j[0]),",", len(onehot_encoded1_j[0][0]),"]")

# pre-trained GloVe model 'glove.6B.50d’ (i.e., Wikipedia 2014 + Gigaword 5, 50d output) to
# embed each word as a 50d vector. Represent each text document as a 2D array, save the
# representation of the whole collection as a 3D array (i.e., a cube).

##Pre-trained glove
processed_collection_r = []
for post in resume_list.Desc:
    #tokenize each review
    tokens = nltk.word_tokenize(post.lower())
    joins = " ".join(tokens)
    processed_collection_r.append(joins)

#convert GloVe format to Word2Vec format
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

glove_encoded = []
for com in tokenized_r:
    com_encoded = []
    for w in com:
        if w in model.vocab:
            com_encoded.append(model.wv[w].tolist())
    glove_encoded.append(com_encoded)

# padding
max = len(glove_encoded[0])
for r in glove_encoded:
    if len(r)>max:
        max = len(r)
for i in range(0,len(glove_encoded)):
    if len(glove_encoded[i]) < max:
        zeros = ((max-len(glove_encoded[i])) * [0])
        glove_encoded[i] = [*zeros, *glove_encoded[i]]

with open('GloVe_Word2Vecoutput_r.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(glove_encoded)


##Pre-trained glove for jobs
processed_collection_j = []
for post in job_list.Desc:
    #tokenize each review
    tokens = nltk.word_tokenize(post.lower())
    joins = " ".join(tokens)
    processed_collection_j.append(joins)

#convert GloVe format to Word2Vec format
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

glove_encoded = []
for post in tokenized_j:
    post_encoded = []
    for w in post:
        if w in model.vocab:
            post_encoded.append(model.wv[w].tolist())
    glove_encoded.append(post_encoded)

# padding
max = len(glove_encoded[0])
for r in glove_encoded:
    if len(r)>max:
        max = len(r)
for i in range(0,len(glove_encoded)):
    if len(glove_encoded[i]) < max:
        zeros = ((max-len(glove_encoded[i])) * [0])
        glove_encoded[i] = [*zeros, *glove_encoded[i]]

with open('GloVe_Word2Vecoutput_j.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # Each review is corresponding to one row, each column is corresponding to one item in the vectors.
    writer.writerows(glove_encoded)

