import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import faiss

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    #return SentenceTransformer('all-mpnet-base-v2', device='cuda')

@st.cache_data
def load_data():
    movies = pd.read_csv('movies_with_text_and_ratings.csv')
    return movies

@st.cache_data
def load_embeddings(movies, _model):
    emb_path = "movie_embeddings.npy"
    if os.path.exists(emb_path):
        movie_embeddings = np.load(emb_path)
    else:
        movie_embeddings = _model.encode(movies['text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
        np.save(emb_path, movie_embeddings)
    return movie_embeddings

@st.cache_data
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

st.title("Recomendador de filmes")
st.write("Escreva a descrição de um filme e receba 5 recomendações")

query = st.text_input("Descreva o filme que deseja assistir: ")
if query:
    model = load_model()
    movies = load_data()
    movie_embeddings = load_embeddings(movies, model)
    index = build_faiss_index(movie_embeddings.copy())

    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    k = 20
    scores, indices = index.search(query_emb, k)
    
    # selecione os filmes correspondentes aos indices
    top5 = movies.iloc[indices[0]].copy()
    
    # também selecione os scores correspondentes
    top5['similarity'] = scores[0]
    
    # conversão dos tipos
    top5['similarity'] = pd.to_numeric(top5['similarity'], errors='coerce')
    top5['rating'] = pd.to_numeric(top5['rating'], errors='coerce')
    
    # calcular score final
    top5['score'] = 0.7 * top5['similarity'] + 0.3 * (top5['rating'] / 5)
    
    # ordenar e pegar top 5
    top5 = top5.sort_values('score', ascending=False).head(5).copy()
    top5.reset_index(drop=True, inplace=True)
    
    st.write(top5[['title', 'genres', 'rating', 'score']])
