import json

with open('train_aug.json', 'r') as data_file:
    data = json.load(data_file)

utt = []
intent = []

for i in data:
    utt.extend(i['utt'])
    intent.extend([i['intent']] * len(i['utt']))


# Write to file
with open('data_aug.json', 'w') as outfile:
    json.dump({'utt': utt, 'intent': intent}, outfile)
