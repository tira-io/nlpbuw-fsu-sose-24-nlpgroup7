import json
from itertools import pairwise
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def write_answers(dataset_path: Path, predicted: list) -> None:
    """ Writes the given answers to a file compliant with the datasets format """
    open(Path(dataset_path) / 'truth.jsonl', 'w').writelines([json.dumps(line)+"\n" for line in predicted])

tira = Client()

    # loading train data
text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")
targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")
    # loading validation data (automatically replaced by test data when run on tira)
text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training")
targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training")

# ground_truth = pd.read_json("train/truth.jsonl", lines=True)
# train_data = pd.read_json("train/pairs.jsonl", lines=True)
# test_data = pd.read_json("test/pairs.jsonl", lines=True)

X = pd.DataFrame({"similarity":[],
                         "id":[]})

X_test_output = pd.DataFrame({"similarity":[],
                         "id":[]})
# Y = ground_truth["same"] # uncomment to go quicker
Y_train = []
for index, text_snippets in enumerate(text_train["text"]):
    print(text_snippets)
    # text_snippet1 = text_snippets[0]
    # text_snippet2 = text_snippets[1]
    truth_class = targets_train["generated"][index]
    # print(text_snippet1)
    # print(text_snippet2)
    vectorizer1 = TfidfVectorizer()

    tf_idf_features1 = vectorizer1.fit_transform(text_snippets)
    # print(vectorizer1.get_feature_names_out())

    matrix = ((tf_idf_features1 * tf_idf_features1.T).A)

    dim = 1  # as only one doc in first category
    # TODO ok to only do this over the 2 documents?
    similarity = matrix[dim:, :dim].mean()

    df2 = pd.DataFrame({"similarity": [similarity],
                        "id": [text_train["id"][index]]})

    X = X.append(df2)
    Y_train.append(truth_class)

    # print(matrix)
    #print(similarity)
    # print(truth_class)
    #if index == 20:
        #break

for index, text_snippets in enumerate(text_validation["text"]):
    # text_snippet1 = text_snippets[0]
    # text_snippet2 = text_snippets[1]

    vectorizer1 = TfidfVectorizer()

    tf_idf_features1 = vectorizer1.fit_transform(text_snippets)
    # print(vectorizer1.get_feature_names_out())

    matrix = ((tf_idf_features1 * tf_idf_features1.T).A)

    dim = 1  # as only one doc in first category
    # TODO ok to only do this over the 2 documents?
    similarity = matrix[dim:, :dim].mean()

    df2 = pd.DataFrame({"similarity": [similarity], "id": [text_validation["id"][index]]})

    X_test_output = X_test_output.append(df2)

print(X)
#X["similarity"]= np.array(X["similarity"]).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y_train, test_size=0.30) #TODO do we have to choose similarity from dataframe?
# Y = Y.astype('float')
# X_train.shape


Lr = LogisticRegression()
X_train_reshape= np.array(X_train["similarity"]).reshape(-1, 1)
X_test_reshape= np.array(X_test["similarity"]).reshape(-1, 1)
Lr.fit(X_train_reshape, y_train)
y_predict = Lr.predict(X_test_reshape)
#df1 = pd.DataFrame({"Actual": y_test, "Predicted": y_predict})

X_test_output_reshape= np.array(X_test_output["similarity"]).reshape(-1, 1)
y_predict_output = Lr.predict(X_test_output_reshape)


predicted = [{'id': x, 'generated': str(y)} for x, y in zip(X_test_output['id'], y_predict_output)]
write_answers("test/", predicted)

print(predicted)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)
