import json
# Load data
with open('data_cleaned.json', 'r') as data_file:
    data = json.load(data_file)

# Augment data so that each intent has at least the same number of samples as the intent with the least number of samples including the augmented data

intent_counts = {}

for i in data["intent"]:
    if i not in intent_counts:
        intent_counts[i] = 0
    intent_counts[i] += 1

with open('data_aug_cleaned.json', 'r') as data_file:
    data_aug = json.load(data_file)

aug_intent_counts = {}

for i in data_aug["intent"]:
    if i not in aug_intent_counts:
        aug_intent_counts[i] = 0
    aug_intent_counts[i] += 1

total_counts = [intent_counts[i] + aug_intent_counts[i] for i in intent_counts]

min_count = min(total_counts)

for i , intent in enumerate(data_aug["intent"]):
    if intent_counts[intent] < min_count:
        data["utt"].append(data_aug["utt"][i])
        data["intent"].append(data_aug["intent"][i])
        intent_counts[intent] += 1


# Sort data by intent

data = sorted(zip(data["utt"], data["intent"]), key=lambda x: x[1])

data = {"utt": [i[0] for i in data], "intent": [i[1] for i in data]}

# Write to file

with open('data_final.json', 'w') as outfile:
    json.dump(data, outfile)
