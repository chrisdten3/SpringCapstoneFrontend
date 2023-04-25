import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import streamlit as st

st.header('Hoya Movies')
st.markdown('Created by the Hoyalytics Analyst Team')
st.markdown("This website uses NLP libraries such as Gensim and sklearn to produce a movie recommender system")

df = pd.read_csv("combined_data.csv")
clean = df[['title', 'genres' ,'overview']]
clean['title'] = clean['title'].str.lower()
clean.head()

movies_df = clean.dropna(subset=["overview"])
movies_df['overview'].isnull().values.any()

def vectorize_overview(overview, model):
    words = overview.lower().split()
    vectors = []
    for word in words:
        if word in model.wv.key_to_index:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(vectors, axis=0)

model = Word2Vec.load('./training.model')

vectors = [vectorize_overview(description, model) for description in movies_df['overview']]

searchedMovie = st.text_input('Find some movies similair to...').lower()
st.markdown('##')
st.subheader('Most Similair Movies: ')

def find_similar_movies(title, movies_df, vectors):
    if title not in movies_df['title'].values:
        print(f"Error: '{title}' not found in movies DataFrame.")
        return None

    index = movies_df[movies_df['title'] == title].index[0]
    similarity_scores = cosine_similarity([vectors[index]], vectors)[0]
    similar_indices = similarity_scores.argsort()[::-1][1:]
    similar_movies = [(movies_df.iloc[i]['title'], similarity_scores[i]) for i in similar_indices]
    return similar_movies


similar_movies = find_similar_movies(searchedMovie, movies_df, vectors)
if similar_movies == None: 
    st.write("Please enter a new movie")
else:
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[:10]
    for movie, score in similar_movies:
        print(movie,score)
        st.write(movie, score)

