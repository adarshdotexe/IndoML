# Use Sentence Transformers and cleanlab to get quality data samples from the augmented data.

import json
import cleanlab
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Load data
with open('data_aug.json', 'r') as data_file:
    data = json.load(data_file)

transformer = SentenceTransformer('llmrails/ember-v1')

text_embeddings = transformer.encode(data["utt"])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(data["intent"])

labels = encoder.transform(data["intent"])

model = LogisticRegression(max_iter=1500)

pred_probs = cross_val_predict(model, text_embeddings, labels, method="predict_proba", n_jobs=-1)

lab = Datalab(data, label_name="intent")
lab.find_issues(pred_probs=pred_probs, features=text_embeddings)

# Remove all the samples that are classified as issues in lab report
# and write the cleaned data to file

# is_label_issue is_outlier_issue  is_near_duplicate_issue is_non_iid_issue if either of these is true, then it is an issue

issue_indices = [i for i in range(len(data["utt"])) if lab.issues["is_label_issue"][i] or lab.issues["is_outlier_issue"][i] or lab.issues["is_near_duplicate_issue"][i] or lab.issues["is_non_iid_issue"][i]]

cleaned_data = {"utt": [], "intent": []}

for i in range(len(data["utt"])):
    if i not in issue_indices:
        cleaned_data["utt"].append(data["utt"][i])
        cleaned_data["intent"].append(data["intent"][i])

with open('data_cleaned.json', 'w') as outfile:
    json.dump(cleaned_data, outfile)
