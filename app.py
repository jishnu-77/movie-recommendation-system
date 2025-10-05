from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

app = Flask(__name__, template_folder="../templates")

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '..', 'app')
movies = pd.read_csv(os.path.join(data_path, 'movies.csv'))
ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
df = pd.merge(ratings, movies, on="movieId")

# Preprocess genres
movies['genres'] = movies['genres'].fillna('')

# TF-IDF for genres (Content-Based)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Collaborative Filtering using SVD
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD()
svd.fit(trainset)

# Home page
@app.route("/")
def index():
    return render_template("dashboard.html")

# Hybrid Recommendation Route
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_id = int(request.form["user_id"])
        movie_title = request.form["movie_title"]

        if movie_title not in indices:
            return render_template("dashboard.html", error="Movie not found!")

        idx = indices[movie_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]
        movie_indices = [i[0] for i in sim_scores]
        similar_movies = movies.iloc[movie_indices][['movieId', 'title']]
        similar_movies['predicted_rating'] = similar_movies['movieId'].apply(
            lambda x: svd.predict(user_id, x).est
        )
        top_recommendations = similar_movies.sort_values('predicted_rating', ascending=False).head(10)
        return render_template("dashboard.html", recommendations=top_recommendations['title'].tolist())

    except Exception as e:
        return render_template("dashboard.html", error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
