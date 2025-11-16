import streamlit as st
import os
import numpy as np
import pandas as pd 
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2', device='cuda')

@st.cache_data
def load_data():
    movies = pd.read_csv('movies_with_text_and_ratings.csv')
    return movies

# normalizar vetores
def normalize_l2(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # evita divisao por zero 
    return embeddings / norms

@st.cache_data
def load_embeddings(movies, _model):
    emb_path = "movie_embeddings.npy"
    if os.path.exists(emb_path):
        movie_embeddings = np.load(emb_path)
    else:
        movie_embeddings = _model.encode(movies['text'].tolist(), 
                                         show_progress_bar=True, 
                                         convert_to_numpy=True)
        np.save(emb_path, movie_embeddings)
        
    movie_embeddings_normalized = normalize_l2(movie_embeddings)
    return movie_embeddings_normalized

st.title("Recomendador de filmes")
st.write("Escreva a descrição de um filme e receba 5 recomendações")

query = st.text_input("Descreva o filme que deseja assistir: ")
if query:
    model = load_model()
    movies = load_data()
    
    movie_embeddings = load_embeddings(movies, model)

    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb_normalized = normalize_l2(query_emb) # Use the same helper

    k = 20
    
    all_scores = np.dot(movie_embeddings, query_emb_normalized.T).flatten()
    
    top_k_indices = np.argsort(all_scores)[::-1][:k]
    
    top_k_scores = all_scores[top_k_indices]
    
    top5 = movies.iloc[top_k_indices].copy()
    
    top5['similarity'] = top_k_scores
    
    top5['similarity'] = pd.to_numeric(top5['similarity'], errors='coerce')
    top5['rating'] = pd.to_numeric(top5['rating'], errors='coerce')
    
    top5['score'] = 0.7 * top5['similarity'] + 0.3 * (top5['rating'] / 5)
    
    top5 = top5.sort_values('score', ascending=False).head(5).copy()
    top5.reset_index(drop=True, inplace=True)
    
    st.write(top5[['title', 'genres', 'rating', 'score']])
