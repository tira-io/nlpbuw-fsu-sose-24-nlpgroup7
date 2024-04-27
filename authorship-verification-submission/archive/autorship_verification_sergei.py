from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory


# calculate number of words
def text_length(text):
    return len(text.split())

# calculate average sentence length
def av_sentence_length(text):
    sentences = text.split('.')
    return sum(len(x.split()) for x in sentences) / len(sentences)

# count bigrams
def count_ngrams(text, n = 2):
    words = word_tokenize(text)
    return len(list(nltk.ngrams(words, n)))

def count_unique_words(text):
    words = word_tokenize(text)
    return len(set(words))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# batch gradient descent with logistic regression
def BGD(X, c, itr, learning_rate=0.1):
    t = 0
    n, p = X.shape
    # inititalize random weights
    w = np.random.rand(p, 1)
    eps = 0.01 # loss bound
    global_loss = np.inf
    while (global_loss >= eps) and (t <= itr):
        loss = 0
        # initialize weight incremnet
        delta_w = np.zeros((p, 1))
        t += 1
        for j in range(n):
            x = X[:][j].reshape(-1,1)
            z = np.dot(np.transpose(w),x)
            y = sigmoid(z)
            delta = c[j] - y
            loss += delta * x # logistic loss increment
            delta_w += learning_rate * loss
            # print(f"Iteration {t}, Sample {j}, z={z}, y={y}, delta={delta}, loss={loss}, delta_w={delta_w}")
        w += delta_w
        global_loss = np.linalg.norm(loss)
        print(f"Iteration: {t}, Global loss: {global_loss}")
    return w

def preprocess_training_data(text_train, targets_train):
    features = pd.DataFrame({})
    features["id"] = text_train["id"]
    features["length"] = text_train["text"].apply(text_length)
    features["length_normalized"] = (features["length"]-features["length"].min()) / (features["length"].max()-features["length"].min())
    features["av_sentence_length"] = text_train["text"].apply(av_sentence_length)
    features["av_sentence_length_normalized"] = (features["av_sentence_length"]-features["av_sentence_length"].min()) / (features["av_sentence_length"].max()-features["av_sentence_length"].min())
    features["ngrams"] = text_train["text"].apply(count_ngrams)
    features["ngrams_normalized"] = (features["ngrams"]-features["ngrams"].min()) / (features["ngrams"].max()-features["ngrams"].min())
    features["unique_words"] = text_train["text"].apply(count_unique_words) / features["length"]

    features = features.merge(targets_train[['id', 'generated']], on='id', how='left')
    cleaned_features = features.drop(['id','length', 'av_sentence_length', 'ngrams'], axis=1).dropna()
    
    means = np.array(cleaned_features.loc[:,:"unique_words"].mean())
    stds = np.array(np.std(cleaned_features.loc[:,:"unique_words"], axis=0))
    
    X = cleaned_features.loc[:,:"unique_words"].to_numpy()
    c = cleaned_features.loc[:,"generated"].to_numpy()
    
    X_standardized = (X - means) / stds
    
    return X_standardized, c

def preprocess_test_data(text_validation, targets_validation):
    features_test = pd.DataFrame({})
    features_test["id"] = text_validation["id"]
    features_test["length"] = text_validation["text"].apply(text_length)
    features_test["length_normalized"] = (features_test["length"]-features_test["length"].min()) / (features_test["length"].max()-features_test["length"].min())
    features_test["av_sentence_length"] = text_validation["text"].apply(av_sentence_length)
    features_test["av_sentence_length_normalized"] = (features_test["av_sentence_length"]-features_test["av_sentence_length"].min()) / (features_test["av_sentence_length"].max()-features_test["av_sentence_length"].min())
    features_test["ngrams"] = text_validation["text"].apply(count_ngrams)
    features_test["ngrams_normalized"] = (features_test["ngrams"]-features_test["ngrams"].min()) / (features_test["ngrams"].max()-features_test["ngrams"].min())
    features_test["unique_words"] = text_validation["text"].apply(count_unique_words) / features_test["length"]

    features_test = features_test.merge(targets_validation[['id', 'generated']], on='id', how='left')

    cleaned_features_test = features_test.drop(['id','length', 'av_sentence_length', 'ngrams'], axis=1).dropna()

    X_test = cleaned_features_test.loc[:,:"unique_words"].to_numpy()

    means_test = np.array(cleaned_features_test.loc[:,:"unique_words"].mean())
    stds_test = np.array(np.std(cleaned_features_test.loc[:,:"unique_words"], axis=0))

    X_standardized_test = (X_test - means_test) / stds_test
    c_test = cleaned_features_test.loc[:, "generated"].to_numpy()
    
    return X_standardized_test, c_test

# calculate the label scores
def predict(X, w):
    z = np.dot(X, w)
    probabilities = sigmoid(z)  
    return probabilities

# classify the scores
def classify(probabilities, threshold=0.5):
    return (probabilities >= threshold).astype(int)

def train_classifier(training_data, training_labels):
    X_standardized, c = preprocess_training_data(training_data, training_labels)
    w = BGD(X_standardized, c, 100, 0.0000015)
    return w

    


if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    w = train_classifier(text_train, targets_train)
    
    X_standardized_test, c_test = preprocess_test_data(text_validation, targets_validation)
    probabilities = predict(X_standardized_test, w)
    labels = classify(probabilities, threshold=0.5)

    # print("Predicted probabilities:", probabilities)
    # print("Class labels:", labels)
    
    labels = labels.flatten()
    incorrect_predictions = np.sum(labels != c_test)
    misclassification_rate = incorrect_predictions/len(labels)
    print(f"Total examples: {len(c_test)}, Incorrect predictions: {incorrect_predictions}, Misclassification rate: {misclassification_rate}")