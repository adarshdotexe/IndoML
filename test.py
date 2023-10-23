import json

with open('surprise.data', 'r') as data_file:
    data = [json.loads(line) for line in data_file]
print(len(data))
with open('surprise.solution', 'r') as solution_file:
    solutions = [json.loads(line) for line in solution_file]
intent_map = {item['indoml_id']: item['intent'] for item in solutions}
indoml_ids = [item['indoml_id'] for item in data]
intent = [intent_map[indoml_id] for indoml_id in indoml_ids]
utt = [item['utt'] for item in data]

# Write to file
with open('data.json', 'w') as outfile:
    json.dump({'utt': utt, 'intent': intent}, outfile)