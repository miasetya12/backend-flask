from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson.json_util import dumps
from bson.objectid import ObjectId  # Import ObjectId to handle MongoDB Object IDs
from gensim.models import Word2Vec
# from gensim.models import FastText
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.model_selection import train_test_split


from flask_cors import CORS # type: ignore

app = Flask(__name__)

# Koneksi ke MongoDB Atlas
client = MongoClient("mongodb+srv://miasetyautami:20xO81m3RtBrqBZr@cluster1.ohirn.mongodb.net/")
db = client['makeup_product']
collection = db['desc_product_full']
ratings_collection = db['review_product']
products_collection = db['desc_product_full']

@app.route('/products', methods=['GET'])
def get_products():
    """Fetch all product descriptions from MongoDB."""
    try:
        products = collection.find().sort("price", -1).limit(20)  # Sort by price in ascending order
        return dumps(products)  # Use dumps for JSON serialization
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/products/<int:product_id>', methods=['GET'])  # Mengubah tipe parameter ke int
def get_product(product_id):
    """Fetch a product description by ID from MongoDB."""
    try:
        print(f"Received product_id: {product_id}")  # Debugging line
        product = collection.find_one({"product_id": product_id})  # Query menggunakan product_id sebagai Int
        if product:
            print(product) 
            return dumps(product)  # Serialize the product data
        else:
            return jsonify({"error": "Product not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def top_n_recommendations_unique(recommendations, target_makeup_type, target_makeup_part, top_n):
    # Print target makeup type
    print(target_makeup_part)
    print(target_makeup_type)

    # Sort based on available score columns
    if 'final_score' in recommendations.columns:
        sorted_recommendations = recommendations.sort_values(by='final_score', ascending=False)
    elif 'score_svd' in recommendations.columns:
        sorted_recommendations = recommendations.sort_values(by='score_svd', ascending=False)
    elif 'score' in recommendations.columns:
        sorted_recommendations = recommendations.sort_values(by='score', ascending=False)
    else:
        sorted_recommendations = recommendations  # No sorting if neither 'score' nor 'score_svd' exists

    # Get all unique product names
    unique_product_names = set()  # Set to keep track of unique product names
    unique_recommendations = []  # List to store unique recommendations

    for _, row in sorted_recommendations.iterrows():
        product_name = row['product_name']
        # Check if the product name is already in the set
        if product_name not in unique_product_names:
            unique_product_names.add(product_name)  # Add to set
            unique_recommendations.append(row)  # Append row to the list

    # Convert unique recommendations to DataFrame
    unique_recommendations_df = pd.DataFrame(unique_recommendations)
    print("Total unique recommendations:", len(unique_recommendations_df))

    makeup_part_len = unique_recommendations_df[unique_recommendations_df['makeup_part'] == target_makeup_part]
    print("Total unique target_makeup_part-:", len(makeup_part_len))

    makeup_type_len = unique_recommendations_df[unique_recommendations_df['makeup_type'] == target_makeup_type]
    print("Total unique target_makeup_type:", len(makeup_type_len))

    # If the number of unique recommendations is less than top_n, filter by makeup_part
    if len(makeup_type_len) < 50:
        print(f"Unique recommendations less than {top_n}. Using makeup_part: {target_makeup_part}")
        unique_recommendations_df = unique_recommendations_df[unique_recommendations_df['makeup_part'] == target_makeup_part]
        print("Total unique target_makeup_part:", len(unique_recommendations_df))

    # If the number of unique recommendations is more than top_n, filter by makeup_type
    elif len(makeup_type_len) > 50:
        print(f"Unique recommendations more than {top_n}. Using makeup_type: {target_makeup_type}")
        unique_recommendations_df = unique_recommendations_df[unique_recommendations_df['makeup_type'] == target_makeup_type]
        print("Total unique target_makeup_type:", len(unique_recommendations_df))

    # Get the top-N recommendations
    top_n_recommendations_df = unique_recommendations_df.head(top_n)
    return top_n_recommendations_df

##TFIDF
def cbf_tfidf(target_product_id, user_id, skin_type='', skin_tone='', under_tone=''):
    # Convert MongoDB collections to pandas DataFrames
    data = pd.DataFrame(list(ratings_collection.find()))
    products = pd.DataFrame(list(products_collection.find()))

    # Ensure 'unique_data_clean' column is not empty
    products['unique_data_clean'] = products['unique_data_clean'].astype(str).fillna('')

    # Initialize the TF-IDF Vectorizer
    tfidf = TfidfVectorizer()

    # Generate TF-IDF matrix for product descriptions
    tfidf_matrix = tfidf.fit_transform(products['unique_data_clean'])

    # Find the target product
    target_product_row = products[products['product_id'] == target_product_id]
    if target_product_row.empty:
        return [], None, None

    # Generate target product description with additional info
    target_product_description = target_product_row['unique_data_clean'].values[0]
    target_makeup_part = target_product_row['makeup_part'].values[0]  # Get target makeup_part
    target_makeup_type = target_product_row['makeup_type'].values[0]

    # Append skin and makeup type if provided to target description
    if skin_type:
        target_product_description += f" {skin_type}"
    if skin_tone:
        target_product_description += f" {skin_tone}"
    if under_tone:
        target_product_description += f" {under_tone}"
    target_tfidf = tfidf.transform([target_product_description])

    # Identify rated and unrated products
    all_items = products['product_id'].unique()
    user_ratings = data[data['user_id'] == user_id]
    rated_items = user_ratings['product_id'].unique()
    unrated_items = [item for item in all_items if item not in rated_items]

    cosine_sim = cosine_similarity(target_tfidf, tfidf_matrix)
    similarity_scores = list(enumerate(cosine_sim[0]))
    sorted_similar_items = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Prepare filtered recommendations for unrated items
    similar_products_filtered = [
        {
            "product_id": int(products['product_id'].iloc[i]),
            "product_name": products['product_name'].iloc[i],
            "makeup_part": products['makeup_part'].iloc[i],
            "makeup_type": products['makeup_type'].iloc[i],
            "score": float(score)
        }
        for i, score in sorted_similar_items if products['product_id'].iloc[i] in unrated_items
    ]

    # Convert to DataFrame and filter by makeup part
    similar_products_filtered_df = pd.DataFrame(similar_products_filtered, columns=['product_id', 'product_name', 'makeup_part', 'makeup_type', 'score'])
    
    return similar_products_filtered_df, target_makeup_part, target_makeup_type


@app.route('/recommend/tfidf', methods=['GET'])
def recommend_tfidf():
    # Get parameters from query string
    target_product_id = request.args.get('target_product_id', type=int)
    user_id = request.args.get('user_id', type=int)
    skin_type = request.args.get('skin_type', default='', type=str)
    skin_tone = request.args.get('skin_tone', default='', type=str)
    under_tone = request.args.get('under_tone', default='', type=str)
    top_n = request.args.get('top_n', default=10, type=int)  # Added top_n parameter

    # Call the recommendation function
    similar_products, target_makeup_part, target_makeup_type = cbf_tfidf(target_product_id, user_id, skin_type, skin_tone, under_tone)

    # Get the top N unique recommendations
    recommendations_df = top_n_recommendations_unique(similar_products, target_makeup_type, target_makeup_part, top_n)

    # Create response
    response = {
        'recommendations': recommendations_df.to_dict(orient='records'),
        'makeup_part': target_makeup_part,
        'makeup_type': target_makeup_type
    }

    return jsonify(response)

##WORD2VEC
# word2vec_model_path = "model\word2vec_model_sg_e20.model"
# word2vec_model = Word2Vec.load(word2vec_model_path)

# def cbf_word2vec(target_product_id, user_id, skin_type='', skin_tone='', under_tone=''):
#       # Convert MongoDB collections to pandas DataFrames
#     data = pd.DataFrame(list(ratings_collection.find()))
#     products = pd.DataFrame(list(products_collection.find()))

#     # Ensure 'unique_data_clean' column is not empty
#     products['unique_data_clean'] = products['unique_data_clean'].astype(str).fillna('')

#     # Tokenize the data
#     tokenized_data = products['unique_data_clean'].apply(lambda x: x.split() if x else [])

#     # Generate product vectors
#     product_vectors = []
#     for tokens in tokenized_data:
#         valid_tokens = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
#         if valid_tokens:
#             vector = np.mean(valid_tokens, axis=0)
#         else:
#             vector = np.zeros(word2vec_model.vector_size)
#         product_vectors.append(vector)

#     # Find the target product
#     target_product_row = products[products['product_id'] == target_product_id]
#     if target_product_row.empty:
#         return pd.DataFrame(), None, None

#     target_product_description = target_product_row['unique_data_clean'].values[0]
#     target_makeup_part = target_product_row['makeup_part'].values[0]  # Get target makeup_part
#     target_makeup_type = target_product_row['makeup_type'].values[0]

#     # Append skin and makeup type if provided
#     if skin_type:
#         target_product_description += f" {skin_type}"
#     if skin_tone:
#         target_product_description += f" {skin_tone}"
#     if under_tone:
#         target_product_description += f" {under_tone}"

#     # Generate target vector
#     target_tokens = target_product_description.split() if target_product_description else []
#     valid_target_tokens = [word2vec_model.wv[token] for token in target_tokens if token in word2vec_model.wv]
#     if valid_target_tokens:
#         target_vector = np.mean(valid_target_tokens, axis=0)
#     else:
#         target_vector = np.zeros(word2vec_model.vector_size)

#     # Identify rated and unrated products
#     all_items = products['product_id'].unique()
#     user_ratings = data[data['user_id'] == user_id]
#     rated_items = user_ratings['product_id'].unique()
#     unrated_items = [item for item in all_items if item not in rated_items]

#     # Calculate cosine similarity between target product and all others
#     cosine_sim = cosine_similarity([target_vector], product_vectors)[0]

#     # Sort similarity scores
#     sorted_similar_items = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)

#     # Prepare filtered recommendations
#     similar_products_filtered = [
#         {
#             "product_id": int(products['product_id'].iloc[i]),  # Use product_id based on index
#             "product_name": products['product_name'].iloc[i],
#             "makeup_part": products['makeup_part'].iloc[i],
#             "makeup_type": products['makeup_type'].iloc[i],
#             "score": float(score)
#         }
#         for i, score in sorted_similar_items if products['product_id'].iloc[i] in unrated_items
#     ]

#     # Convert to DataFrame and filter by makeup part
#     similar_products_filtered_df = pd.DataFrame(similar_products_filtered)
#     filtered_recommendations = similar_products_filtered_df[similar_products_filtered_df['makeup_part'] == target_makeup_part]

#     return filtered_recommendations, target_makeup_part, target_makeup_type

# @app.route('/recommend/word2vec', methods=['GET'])
# def recommend_word2vec():
#     # Get parameters from query string
#     target_product_id = request.args.get('target_product_id', type=int)
#     user_id = request.args.get('user_id', type=int)
#     skin_type = request.args.get('skin_type', default='', type=str)
#     skin_tone = request.args.get('skin_tone', default='', type=str)
#     under_tone = request.args.get('under_tone', default='', type=str)
#     top_n = request.args.get('top_n', default='', type=int)  # Add top_n parameter

#     # Call the recommendation function
#     recommendations_df, target_makeup_part, target_makeup_type = cbf_word2vec(target_product_id, user_id, skin_type, skin_tone, under_tone)

#     recommendations_df = top_n_recommendations_unique(recommendations_df, target_makeup_type, target_makeup_part, top_n)

#     # Create response
#     response = {
#         'recommendations': recommendations_df.to_dict(orient='records'),
#         'makeup_part': target_makeup_part,
#         'makeup_type': target_makeup_type
#     }

#     return jsonify(response)

#FASTTEXT
# fasttext_model_path = "model/fasttext_model_sg_e20.model"
# fasttext_model = FastText.load(fasttext_model_path)

# def cbf_fasttext(target_product_id, user_id, skin_type='', skin_tone='', under_tone=''):
#        # Convert MongoDB collections to pandas DataFrames
#     data = pd.DataFrame(list(ratings_collection.find()))
#     products = pd.DataFrame(list(products_collection.find()))

#     products['unique_data_clean'] = products['unique_data_clean'].astype(str).fillna('')

#     # Tokenize the data
#     tokenized_data = products['unique_data_clean'].apply(lambda x: x.split() if x else [])

#     # Generate product vectors
#     product_vectors = []
#     for tokens in tokenized_data:
#         valid_tokens = [fasttext_model.wv[token] for token in tokens if token in fasttext_model.wv]
#         if valid_tokens:
#             vector = np.mean(valid_tokens, axis=0)
#         else:
#             vector = np.zeros(fasttext_model.vector_size)
#         product_vectors.append(vector)

#     # Find the target product
#     target_product_row = products[products['product_id'] == target_product_id]
#     if target_product_row.empty:
#         return pd.DataFrame(), None, None

#     target_product_description = target_product_row['unique_data_clean'].values[0]
#     target_makeup_part = target_product_row['makeup_part'].values[0]
#     target_makeup_type = target_product_row['makeup_type'].values[0]

#     # Append skin and makeup type if provided
#     if skin_type:
#         target_product_description += f" {skin_type}"
#     if skin_tone:
#         target_product_description += f" {skin_tone}"
#     if under_tone:
#         target_product_description += f" {under_tone}"

#     # Generate target vector
#     target_tokens = target_product_description.split() if target_product_description else []
#     valid_target_tokens = [fasttext_model.wv[token] for token in target_tokens]

#     if valid_target_tokens:
#         target_vector = np.mean(valid_target_tokens, axis=0)
#     else:
#         target_vector = np.zeros(fasttext_model.vector_size)  # This case is rare with FastText

#     # Identify rated and unrated products
#     all_items = products['product_id'].unique()
#     user_ratings = data[data['user_id'] == user_id]
#     rated_items = user_ratings['product_id'].unique()
#     unrated_items = [item for item in all_items if item not in rated_items]

#     # Calculate cosine similarity between target product and all others
#     cosine_sim = cosine_similarity([target_vector], product_vectors)[0]

#     # Sort similarity scores
#     sorted_similar_items = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)

#     # Prepare filtered recommendations
#     similar_products_filtered = [
#         {
#             "product_id": int(products['product_id'].iloc[i]),  # Use product_id based on index
#             "product_name": products['product_name'].iloc[i],
#             "makeup_part": products['makeup_part'].iloc[i],
#             "makeup_type": products['makeup_type'].iloc[i],
#             "score": float(score)
#         }
#         for i, score in sorted_similar_items if products['product_id'].iloc[i] in unrated_items
#     ]

#     # Convert to DataFrame and filter by makeup part
#     similar_products_filtered_df = pd.DataFrame(similar_products_filtered)
#     filtered_recommendations = similar_products_filtered_df[similar_products_filtered_df['makeup_part'] == target_makeup_part]

#     return filtered_recommendations, target_makeup_part, target_makeup_type

# @app.route('/recommend/fasttext', methods=['GET'])
# def recommend_fasttext():
#     # Get parameters from query string
#     target_product_id = request.args.get('target_product_id', type=int)
#     user_id = request.args.get('user_id', type=int)
#     skin_type = request.args.get('skin_type', default='', type=str)
#     skin_tone = request.args.get('skin_tone', default='', type=str)
#     under_tone = request.args.get('under_tone', default='', type=str)
#     top_n = request.args.get('top_n', default='', type=int)  # Add top_n parameter

#     # Call the recommendation function
#     recommendations_df, target_makeup_part, target_makeup_type = cbf_fasttext(target_product_id, user_id, skin_type, skin_tone, under_tone)

#     recommendations_df = top_n_recommendations_unique(recommendations_df, target_makeup_type, target_makeup_part, top_n)

#     # Create response
#     response = {
#         'recommendations': recommendations_df.to_dict(orient='records'),
#         'makeup_part': target_makeup_part,
#         'makeup_type': target_makeup_type
#     }

#     return jsonify(response)





if __name__ == "__main__":
    app.run(debug=True)
CORS(app)
