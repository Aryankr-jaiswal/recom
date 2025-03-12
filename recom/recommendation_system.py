import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API configuration
TMDB_API_KEY = "1d9b898a5b7f9ac3dd385c2651f6840e"  # TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

class ContentBasedRecommender:
    def __init__(self):
        # Sample movie data with TMDB IDs
        self.movies_data = {
            'MovieID': list(range(1, 11)),
            'Title': [
                'The Dark Knight',
                'Inception',
                'Interstellar',
                'The Matrix',
                'Avatar',
                'Pulp Fiction',
                'The Shawshank Redemption',
                'Forrest Gump',
                'The Godfather',
                'Fight Club'
            ],
            'Genre': [
                'Action, Crime, Drama',
                'Action, Adventure, Sci-Fi',
                'Adventure, Drama, Sci-Fi',
                'Action, Sci-Fi',
                'Action, Adventure, Fantasy',
                'Crime, Drama',
                'Drama',
                'Drama, Romance',
                'Crime, Drama',
                'Drama'
            ],
            'Description': [
                'Batman fights against organized crime in Gotham City',
                'A thief enters dreams to steal information',
                'Astronauts travel through a wormhole in search of a new home',
                'A computer programmer discovers the truth about artificial reality',
                'A paraplegic marine dispatched to a moon on a unique mission',
                'Various interconnected stories of criminal life in Los Angeles',
                'A banker is sentenced to life in prison for a crime he did not commit',
                'A slow-witted but kind-hearted man experiences several defining historical events',
                'The aging patriarch of an organized crime dynasty transfers control to his son',
                'An insomniac office worker forms an underground fight club'
            ],
            'TMDB_ID': [155, 27205, 157336, 603, 19995, 680, 278, 13, 238, 550],
            'Poster_URL': [None] * 10
        }
        self.movies_df = pd.DataFrame(self.movies_data)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None

    def fetch_movie_posters(self):
        """Fetch movie poster URLs from TMDB API"""
        poster_urls = []
        for tmdb_id in self.movies_df['TMDB_ID']:
            try:
                response = requests.get(
                    f"{TMDB_BASE_URL}/movie/{tmdb_id}",
                    params={"api_key": TMDB_API_KEY}
                )
                if response.status_code == 200:
                    movie_data = response.json()
                    poster_path = movie_data.get('poster_path')
                    if poster_path:
                        poster_urls.append(f"{TMDB_IMAGE_BASE_URL}{poster_path}")
                    else:
                        poster_urls.append(None)
                else:
                    poster_urls.append(None)
            except Exception as e:
                print(f"Error fetching poster for movie {tmdb_id}: {str(e)}")
                poster_urls.append(None)
        return poster_urls

    def preprocess_data(self):
        # Combine genre and description for better content analysis
        self.movies_df['Content'] = self.movies_df['Genre'] + ' ' + self.movies_df['Description']
        
        # Fetch and add movie posters
        self.movies_df['Poster_URL'] = self.fetch_movie_posters()
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df['Content'])
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, movie_title, n_recommendations=3):
        # Find the index of the movie
        idx = self.movies_df[self.movies_df['Title'] == movie_title].index[0]
        
        # Get similarity scores for the movie
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommended_movies = self.movies_df.iloc[movie_indices]
        return recommended_movies[['Title', 'Genre', 'Description', 'Poster_URL']]

def main():
    # Initialize and train the recommender
    recommender = ContentBasedRecommender()
    recommender.preprocess_data()
    
    # Get recommendations for a movie
    movie_title = 'The Dark Knight'
    print(f"\nRecommendations for {movie_title}:")
    recommendations = recommender.get_recommendations(movie_title)
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")

if __name__ == '__main__':
    main()