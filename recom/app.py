from flask import Flask, render_template, request, jsonify
from recommendation_system import ContentBasedRecommender

app = Flask(__name__)
recommender = ContentBasedRecommender()
recommender.preprocess_data()

@app.route('/')
def home():
    return render_template('index.html', movies=recommender.movies_df['Title'].tolist())

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form.get('movie')
    if not movie_title:
        return jsonify({'error': 'No movie title provided'}), 400
    
    try:
        recommendations = recommender.get_recommendations(movie_title)
        recommended_movies = []
        for _, movie in recommendations.iterrows():
            recommended_movies.append({
                'title': movie['Title'],
                'genre': movie['Genre'],
                'description': movie['Description'],
                'poster_url': movie['Poster_URL']
            })
        return jsonify({
            'success': True,
            'recommendations': recommended_movies
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)