# ========== vectorize.py (đã chỉnh sửa để trả về vectorizer fitted) ==========
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec as W2V, FastText as FT
import numpy as np
import pandas as pd
import spacy

def OnehotEncoding(sentences):
    
    nlp = spacy.load("en_core_web_sm")
    tokens = [token.text for s in sentences for token in nlp(s) if not token.is_stop and not token.is_punct]
    encoded = OneHotEncoder(sparse_output=False)
    result = encoded.fit_transform(np.array(tokens).reshape(-1, 1))
    return pd.DataFrame(result, columns=encoded.get_feature_names_out()), encoded

def BagofWord(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()), vectorizer

def BagofN_Gram(sentences):
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform(sentences)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()), vectorizer

def TF_IDF(sentences):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()), vectorizer

def Word2Vec(sentences):
    nlp = spacy.load("en_core_web_sm")
    tokenized = [[token.text for token in nlp(sent)] for sent in sentences]
    model = W2V(tokenized, vector_size=100, window=5, min_count=1)
    vectors = [np.mean([model.wv[word] for word in sent if word in model.wv] or [np.zeros(100)], axis=0) for sent in tokenized]
    return pd.DataFrame(vectors), model

def FastText(sentences):
    nlp = spacy.load("en_core_web_sm")
    tokenized = [[token.text for token in nlp(sent)] for sent in sentences]
    model = FT(tokenized, vector_size=100, window=5, min_count=1)
    vectors = [np.mean([model.wv[word] for word in sent if word in model.wv] or [np.zeros(100)], axis=0) for sent in tokenized]
    return pd.DataFrame(vectors), model
