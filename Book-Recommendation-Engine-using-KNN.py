#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load datasets
books_path = 'BX-Books.csv'
ratings_path = 'BX-Book-Ratings.csv'
users_path = 'BX-Users.csv'

books_df = pd.read_csv(
    books_path,
    sep=';',
    encoding='ISO-8859-1',
    usecols=['bookId', 'title', 'author'],
    names=['bookId', 'title', 'author'],
    header=0,
    dtype={'bookId': str, 'title': str, 'author': str}
)

ratings_df = pd.read_csv(
    ratings_path,
    sep=';',
    encoding='ISO-8859-1',
    usecols=['userId', 'bookId', 'rating'],
    names=['userId', 'bookId', 'rating'],
    header=0,
    dtype={'userId': 'int32', 'bookId': str, 'rating': 'float32'}
)

# Filter active users (more than 200 ratings)
user_rating_counts = ratings_df.groupby('userId')['rating'].count().reset_index(name='rating_count')
active_users = user_rating_counts[user_rating_counts['rating_count'] > 200]['userId'].tolist()

# Merge ratings with book metadata
merged_df = pd.merge(ratings_df, books_df, on='bookId')

# Count ratings per book title
book_rating_counts = merged_df.groupby('title')['rating'].count().reset_index(name='total_rating_count')
rating_data = pd.merge(merged_df, book_rating_counts, on='title')

# Filter books with more than 100 ratings and users with > 200 ratings
popular_books = rating_data[
    (rating_data['total_rating_count'] > 100) &
    (rating_data['userId'].isin(active_users))
]

# Create pivot table and matrix
ratings_matrix = popular_books.pivot_table(index='title', columns='userId', values='rating', aggfunc='mean').fillna(0)
ratings_csr = csr_matrix(ratings_matrix.values)

# Book recommendation function
def get_recommends(book_title=""):
    knn_model = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=5)
    knn_model.fit(ratings_csr)

    try:
        book_idx = ratings_matrix.index.get_loc(book_title)
    except KeyError:
        return [book_title, []]  # Book not found

    distances, indices = knn_model.kneighbors(ratings_matrix.iloc[book_idx, :].values.reshape(1, -1))
    recommendations = [
        [ratings_matrix.index[idx], dist]
        for idx, dist in zip(indices.flatten()[1:], distances.flatten()[1:])
    ]

    return [book_title, recommendations[::-1]]  # Reverse for same order

# Test function
def test_book_recommendation():
    result = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    print(result)
    print()

    expected_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
    expected_distances = [0.8, 0.77, 0.77, 0.77]

    print(expected_books)
    print(expected_distances)

    passed = result[0] == "Where the Heart Is (Oprah's Book Club (Paperback))"

    for i in range(2):  # Check first two
        title, dist = result[1][i]
        if title not in expected_books or abs(dist - expected_distances[i]) >= 0.05:
            passed = False

    if passed:
        print("You passed the challenge! 🎉🎉🎉🎉🎉")
    else:
        print("You haven't passed yet. Keep trying!")

test_book_recommendation()
