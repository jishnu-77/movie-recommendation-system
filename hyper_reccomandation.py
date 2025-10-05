# recommender/hybrid_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
df = pd.merge(ratings, movies, on="movieId")

# TF-IDF
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Collaborative Filtering
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD()
svd.fit(trainset)

def hybrid_recommender(user_id, movie_title, top_n=10):
    if movie_title not in indices:
        return []
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]
    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies.iloc[movie_indices][['movieId', 'title']]
    similar_movies['predicted_rating'] = similar_movies['movieId'].apply(
        lambda x: svd.predict(user_id, x).est
    )
    top_recommendations = similar_movies.sort_values('predicted_rating', ascending=False).head(top_n)
    return top_recommendations['title'].tolist()
