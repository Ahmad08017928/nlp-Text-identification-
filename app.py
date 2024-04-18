import streamlit as st
import pandas as pd
import numpy as np
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

@st.cache_data(max_entries=1)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

@st.cache_data(max_entries=1)
def tokenize(text):
    return text.split()

@st.cache_data(max_entries=1)
def remove_stopwords(text):
    text_str = ' '.join(text)
    return stopword_remover.remove(text_str).split()

@st.cache_data(max_entries=1)
def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

@st.cache_data(max_entries=1)
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def create_features_vector(text, vocabulary):
    vector = [1 if word in text else 0 for word in vocabulary]
    return np.array(vector)

def knn_classify(k, sample_vector, data, labels, vocabulary):
    similarities = []
    for _, row in data.iterrows():
        text_tokens = row['Text Tweet']
        text_vector = create_features_vector(text_tokens, vocabulary)
        similarity = cosine_similarity(sample_vector, text_vector)
        similarities.append((similarity, labels[_]))
    similarities.sort(reverse=True)
    k_nearest = similarities[:k]
    counts = {}
    for _, label in k_nearest:
        counts[label] = counts.get(label, 0) + 1
    majority_label = max(counts, key=counts.get)
    return majority_label

def main():
    st.title("Sentiment Analysis")
    user_input = st.text_area('Please enter the text you want to identify')
    button = st.button("Analyze")

    if user_input and button:
        # Preprocessing
        preprocessed_text = preprocess_text(user_input)
        tokens = tokenize(preprocessed_text)
        clean_tokens = remove_stopwords(tokens)
        stemmed_tokens = stem_tokens(clean_tokens)
        
        # Load data
        data = pd.read_csv('data\dataset_tweet_sentiment_opini_film.csv')
        
        # Calculate TF-IDF
        vocab = set(word for word in ' '.join(data['Text Tweet']).split())
        sample_vector = create_features_vector(stemmed_tokens, vocab)
        
        # Predict sentiment
        k = 3
        predicted_sentiment = knn_classify(k, sample_vector, data, data['Sentiment'], vocab)
        
        # Display result
        st.write("Predicted Sentiment:", predicted_sentiment)

if __name__ == "__main__":
    main()
