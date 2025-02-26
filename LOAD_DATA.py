import pickle

with open('data.pickle', 'rb') as f:
    dataset = pickle.load(f)

data = dataset['data']
labels = dataset['labels']

print(f"✅ Dataset loaded successfully!")
print(f"Total Samples: {len(data)}")
print(f"Unique Labels: {set(labels)}")
