import pandas as pd

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

# Mergeia as tags por filmes
tags_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))) .reset_index()

# Combina tudo em um dataset
movies = movies.merge(tags_agg, on='movieId', how='left')
movies['text'] = (
    movies['title'].fillna('') + ' ' +
    movies['genres'].fillna('') + ' ' +
    movies['tag'].fillna('')
)

# Compute average rating
avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
movies = movies.merge(avg_ratings, on='movieId', how='left')

movies.to_csv('movies_with_text_and_ratings.csv', index=False)