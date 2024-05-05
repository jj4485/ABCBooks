import json
import gensim.downloader as gensim
import random
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).parent
TEXT_FILE = ROOT_DIR / 'words.txt'

# Load data from the JSON file
with open(TEXT_FILE, 'r', encoding='utf-8', errors='ignore') as file:
    data = json.load(file)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Extract a list of words
words = [item['term'] for item in data]
print(words)

# Set a seed for reproducibility
#random.seed(42)

# Randomly select 1000 words
#random_words = random.sample(words, 1000)

# Load the GloVe model
model = gensim.load('glove-wiki-gigaword-100')

# Get embeddings for selected words
embeddings = {word: model[word] for word in words}

# Create a DataFrame from embeddings
df = pd.DataFrame.from_dict(embeddings, orient='index')


# Correctly using the DataFrame's index to get the word labels
labels = df.index
#print(labels)

features = df.iloc[:, :]
#print(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


kmeans = KMeans(n_clusters=3, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

# Now correctly including word labels in the clustered data
clustered_data = pd.DataFrame({'Label': labels, 'Cluster': clusters})
print(clustered_data)
