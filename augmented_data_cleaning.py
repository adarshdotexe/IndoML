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

# Load model
model = SentenceTransformer('llmrails/ember-v1')

# Get embeddings
embeddings = model.encode(data['utt'])

# Get confident learning labels
classifier = LogisticRegression()
classifier.fit(embeddings, data['intent'])

# Get confident learning labels
noise_matrix = cleanlab.latent_estimation.estimate_confident_joint_and_cv_pred_proba(
    embeddings, data['intent'], classifier, pulearn_threshold=0.8
)
