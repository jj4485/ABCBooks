import json
import gensim.downloader as api
import random
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent
TEXT_FILE = ROOT_DIR / 'words.txt'

# Load data from the JSON file
with open(TEXT_FILE, 'r', encoding='utf-8', errors='ignore') as file:
    data = json.load(file)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Extract a list of words
words = [item['term'] for item in data]


# Set a seed for reproducibility
random.seed(42)

# Randomly select 1000 words
random_words = random.sample(words, 1000)

# Load the GloVe model
model = api.load('glove-wiki-gigaword-100')

# Get embeddings for selected words
embeddings = {word: model[word] for word in random_words if word in model}

# Create a DataFrame from embeddings
df = pd.DataFrame.from_dict(embeddings, orient='index')



# Correctly using the DataFrame's index to get the word labels
labels = df.index
###print(labels)

features = df.iloc[:, :]
#print(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


kmeans = KMeans(n_clusters=25, n_init=10)
clusters = kmeans.fit_predict(features_scaled)


clustered_data = pd.DataFrame({'Label': labels, 'Cluster': clusters})
print(clustered_data)

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(features_scaled)

# Plotting the results of t-SNE with cluster labels in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=clusters, cmap='viridis', alpha=0.5)
plt.title('3D t-SNE visualization of word embeddings clustered into 5 groups')
fig.colorbar(scatter)
plt.show() 