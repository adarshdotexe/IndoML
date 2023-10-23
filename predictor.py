# Use sentence transformers to get embeddings for the utterances.
# Train a linear model on the embeddings to predict the intent.

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Load data
with open('data_final.json', 'r') as data_file:
    data = json.load(data_file)

transformer = SentenceTransformer('llmrails/ember-v1')

text_embeddings = transformer.encode(data["utt"])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(data["intent"])

labels = encoder.transform(data["intent"])

model = torch.nn.Sequential(
    torch.nn.GELU(),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(768, 768),
    torch.nn.GELU(),
    torch.nn.Linear(768, 150)
)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

dataset = torch.utils.data.TensorDataset(torch.tensor(text_embeddings), torch.tensor(labels))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(batch[0])
        loss = loss_fn(pred, batch[1])
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())

