import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# Validate and filter data to ensure consistent feature vector lengths
raw_data = data_dict['data']
raw_labels = data_dict['labels']

# Find the expected feature length (most common length)
from collections import Counter
lengths = [len(sample) for sample in raw_data]
most_common_length = Counter(lengths).most_common(1)[0][0]

print(f"Found {len(raw_data)} samples")
print(f"Most common feature vector length: {most_common_length}")
print(f"Filtering samples to ensure consistent length...")

# Filter to only include samples with the expected length
filtered_data = []
filtered_labels = []
for sample, label in zip(raw_data, raw_labels):
    if len(sample) == most_common_length:
        filtered_data.append(sample)
        filtered_labels.append(label)

print(f"After filtering: {len(filtered_data)} samples with consistent length")

if len(filtered_data) == 0:
    print("Error: No valid samples found! Please regenerate the dataset.")
    exit(1)

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
