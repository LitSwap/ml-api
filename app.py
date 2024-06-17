from flask import Flask, request, jsonify, render_template
from flask_swagger_ui import get_swaggerui_blueprint
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np
import requests

app = Flask(__name__)

response = requests.get('https://litswap-project.et.r.appspot.com/booksnoauth')

try:
    response_json = response.json()
except requests.exceptions.JSONDecodeError:
    response_json = []
    print("Response is not in JSON format.")
    print("Status code:", response.status_code)

unique_books = {}
for book in response_json:
    isbn = book['isbn']
    if isbn not in unique_books:
        year = book['year'].split("-")[0] if 'year' in book and book['year'] and book['year'].isdigit() else 'Tahun Tidak Tersedia'
        unique_books[isbn] = {
            'title': book['title'],
            'author': book['author'],
            'genre': book['genre'],
            'year': str(year)  
        }

# Convert the unique_books dictionary to a DataFrame
df = pd.DataFrame.from_dict(unique_books, orient='index')

# Preprocess data
label_encoder_author = LabelEncoder()
label_encoder_genre = LabelEncoder()

df['author_encoded'] = label_encoder_author.fit_transform(df['author'])
df['genre_encoded'] = label_encoder_genre.fit_transform(df['genre'])

df['year'] = df['year'].apply(lambda x: int(x) if x.isdigit() else -1)

features = df[['author_encoded', 'genre_encoded', 'year']]

# Fit k-NN model
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_books():
    data = request.json
    if not data or 'favorite_books' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    favorite_books = data['favorite_books']
    n_recommendations = int(data.get('n_recommendations', 5))  # Ensure n_recommendations is an integer

    try:
        book_indices = [df.index.get_loc(book) for book in favorite_books if book in df.index]
        if not book_indices:
            return jsonify({'error': 'No valid favorite books found'}), 404
    except KeyError:
        return jsonify({'error': 'One or more books not found'}), 404

    mean_feature_vector = np.mean(features.iloc[book_indices, :], axis=0).values.reshape(1, -1)
    distances, indices = knn.kneighbors(mean_feature_vector, n_neighbors=n_recommendations + 1)

    recommended_books = [df.iloc[indices[0][i]].name for i in range(1, len(indices[0]))]

    return jsonify(recommended_books)

# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Book Recommendation API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run(debug=True)
