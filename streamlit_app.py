# ===============================
# Improved Streamlit App: Movie Clustering
# ===============================

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np

st.set_page_config(page_title="Improved Movie Clustering", layout="wide")
st.title("🎬 Movie Tag & Genre-Based Clustering")

# ----------------------------
# Step 1: Load Data
# ----------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")
    # Combine all tags per movie
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    movie_tags = movie_tags.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    movie_tags['tag'] = movie_tags['tag'].str.lower()
    # Combine tags + genres for richer representation
    movie_tags['all_text'] = movie_tags['tag'] + ' ' + movie_tags['genres'].str.replace('|', ' ')
    return movie_tags

movie_tags = load_data()

# ----------------------------
# Step 2: Compute Embeddings
# ----------------------------
@st.cache_data
def embed_movies(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    embeddings = model.encode(texts, device="cpu", show_progress_bar=True)

    return embeddings

X_embeddings = embed_movies(movie_tags['all_text'].tolist())

# ----------------------------
# Step 3: Cluster Movies
# ----------------------------
st.sidebar.header("Clustering Settings")
k = st.sidebar.slider("Number of Clusters (k)", 2, 20, 8)

clustering = AgglomerativeClustering(n_clusters=k)
movie_tags['cluster'] = clustering.fit_predict(X_embeddings)

# ----------------------------
# Step 4: Inspect Clusters
# ----------------------------
st.subheader("Cluster Overview")
cluster_selector = st.sidebar.selectbox("Select Cluster to View", list(range(k)))
cluster_movies = movie_tags[movie_tags['cluster'] == cluster_selector][['title', 'genres']]
st.dataframe(cluster_movies.reset_index(drop=True).head(20))

# ----------------------------
# Step 5: t-SNE Visualization
# ----------------------------
st.subheader("Cluster Visualization (t-SNE)")

@st.cache_data
def tsne_transform(X):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(X)

X_embedded = tsne_transform(X_embeddings)

num_clusters = len(movie_tags['cluster'].unique())
colors = cm.get_cmap('tab20', num_clusters) 
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    hue=movie_tags['cluster'],
    palette=colors,
    legend='full',
    s=50
)
plt.title("t-SNE Visualization of Movie Clusters")
plt.xlabel("")
plt.ylabel("")
plt.xticks([])
plt.yticks([])
st.pyplot(plt)
