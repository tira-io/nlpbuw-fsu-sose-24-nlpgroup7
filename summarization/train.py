from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Load the data
tira = Client()

text_train = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "summarization-train-20240530-training"
).set_index("id")
labels_train = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "summarization-train-20240530-training"
).set_index("id")

text_val= tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
).set_index("id")
labels_val = tira.pd.truths(
    "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
).set_index("id")