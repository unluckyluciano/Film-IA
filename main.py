import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Carichiamo il dataset delle valutazioni e dei film
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Visualizziamo i primi record
print(ratings.head())
print(movies.head())
# Creiamo la matrice utente-film
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)
print(user_movie_matrix.head())