from flask import Flask, request, jsonify, render_template
from flask_swagger_ui import get_swaggerui_blueprint
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('./data/preprocessed_books.csv')

label_encoder_author = LabelEncoder()
label_encoder_publisher = LabelEncoder()

df['book_author_encoded'] = label_encoder_author.fit_transform(df['Book-Author'])
df['publisher_encoded'] = label_encoder_publisher.fit_transform(df['Publisher'])

features = df[['book_author_encoded', 'publisher_encoded', 'Year-Of-Publication']]

# Fit k-NN model
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_books():
    data = request.json
    favorite_books = data['favorite_books']
    n_recommendations = data.get('n_recommendations', 5)
    
    book_indices = [df[df['ISBN'] == book].index[0] for book in favorite_books]
    mean_feature_vector = np.mean(features.iloc[book_indices, :], axis=0).values.reshape(1, -1)
    distances, indices = knn.kneighbors(mean_feature_vector, n_neighbors=n_recommendations + 1)
    
    recommended_books = [df.iloc[indices[0][i]]['ISBN'] for i in range(1, len(indices[0]))]
    
    return jsonify(recommended_books)
# Swagger UI setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Book Recommendation API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    app.run(debug=True)
