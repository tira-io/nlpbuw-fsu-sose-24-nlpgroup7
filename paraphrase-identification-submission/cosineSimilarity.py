from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import numpy as np

nlp = spacy.load('en_core_web_md')

# preprocess sentences by removing punctuation and making everything lowercase
def preprocess(sentence):
    sentence = sentence.lower()
    # get rid of punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence

# compute word vectors for words in sentences
def get_word_vectors(sentence):
    doc = nlp(preprocess(sentence))
    return np.array([token.vector for token in doc])

def calculate_cosine_similarity(sentence1, sentence2):
    word_vectors1 = get_word_vectors(sentence1)
    word_vectors2 = get_word_vectors(sentence2)
    # compute average of all word vectors for each sentence
    embedding1 = np.mean(word_vectors1, axis=0)
    embedding2 = np.mean(word_vectors2, axis=0)
    # compute cosine similarity between averaged embeddings
    return cosine_similarity([embedding1], [embedding2])[0][0]