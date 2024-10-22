from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson.json_util import dumps
from bson.objectid import ObjectId  # Import ObjectId to handle MongoDB Object IDs
from gensim.models import Word2Vec
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.model_selection import train_test_split


from flask_cors import CORS # type: ignore

app = Flask(__name__)

# Koneksi ke MongoDB Atlas
client = MongoClient("mongodb+srv://miasetyautami:20xO81m3RtBrqBZr@cluster1.ohirn.mongodb.net/")
db = client['makeup_product']
collection = db['desc_product_full']


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

def cbf_tfidf(target_product_id, skin_type='', skin_tone='', under_tone='', top_n=''):
    df_produk = pd.DataFrame(list(collection.find()))
    df_produk['unique_data_clean'] = df_produk['unique_data_clean'].astype(str).fillna('')
    
    target_product_row = df_produk[df_produk['product_id'] == target_product_id]
    if target_product_row.empty:
        return [], None  # Return empty list and None if product not found

    target_product_description = target_product_row['unique_data_clean'].values[0]
    target_makeup_type = target_product_row['makeup_part'].values[0]

    # Append skin type, tone, and undertone to the description
    if skin_type:
        target_product_description += f" {skin_type}"
    if skin_tone:
        target_product_description += f" {skin_tone}"
    if under_tone:
        target_product_description += f" {under_tone}"

    df_produk.loc[target_product_row.index, 'unique_data_clean'] = target_product_description
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_produk['unique_data_clean'])
    target_vector = tfidf_matrix[df_produk.index[df_produk['product_id'] == target_product_id][0]]
    cosine_sim = cosine_similarity(target_vector, tfidf_matrix)

    similarity_scores = list(enumerate(cosine_sim[0]))
    sorted_similar_items = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_products_filtered = []
    unique_products = set()  # Use a set to keep track of unique product names

    for i, score in sorted_similar_items:
        product_id = df_produk['product_id'].iloc[i]
        makeup_part = df_produk['makeup_part'].iloc[i]

        if makeup_part == target_makeup_type and product_id != target_product_id:
            product_name = df_produk['product_name'].iloc[i]
            if product_name not in unique_products:  # Check for uniqueness
                unique_products.add(product_name)  # Add to set
                similar_products_filtered.append({
                    "product_id": int(product_id),
                    "product_name": product_name,
                    "makeup_part": makeup_part,
                    "score": float(score)
                })
        
        if len(similar_products_filtered) >= top_n:
            break
    
    # Limit to top_n products
    limited_unique_products = similar_products_filtered[:top_n]

    return limited_unique_products, target_makeup_type

def cbf_word2vec(target_product_id, skin_type='', skin_tone='', under_tone='', top_n=''):
    df_produk = pd.DataFrame(list(collection.find()))
    df_produk['unique_data_clean'] = df_produk['unique_data_clean'].astype(str).fillna('')
    
     # Tokenize the data
    tokenized_data = df_produk['unique_data_clean'].apply(lambda x: x.split() if x else [])

    # Train Word2Vec model
    word2vec_model = Word2Vec(tokenized_data, vector_size=50, window=3, min_count=2, workers=4, sg=False)

    # Generate product vectors
    product_vectors = []
    for tokens in tokenized_data:
        if tokens:
            vector = np.mean([word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv], axis=0)
        else:
            vector = np.zeros(word2vec_model.vector_size)
        product_vectors.append(vector)

    # Find the target product
    target_product_row = df_produk[df_produk['product_id'] == target_product_id]
    if target_product_row.empty:
        return [], None

    target_product_description = target_product_row['unique_data_clean'].values[0]
    target_makeup_type = target_product_row['makeup_part'].values[0]  # Get target makeup_part

    # Append skin and makeup type if provided
    if skin_type:
        target_product_description += f" {skin_type}"
    if skin_tone:
        target_product_description += f" {skin_tone}"
    if under_tone:
        target_product_description += f" {under_tone}"

    # Generate target vector
    target_tokens = target_product_description.split() if target_product_description else []
    if target_tokens:
        target_vector = np.mean([word2vec_model.wv[token] for token in target_tokens if token in word2vec_model.wv], axis=0)
    else:
        target_vector = np.zeros(word2vec_model.vector_size)


    cosine_sim = cosine_similarity([target_vector], product_vectors)
    similarity_scores = list(enumerate(cosine_sim[0]))
    sorted_similar_items = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_products_filtered = []
    unique_products = set()  # Use a set to keep track of unique product names

    for i, score in sorted_similar_items:
        product_id = df_produk['product_id'].iloc[i]
        makeup_part = df_produk['makeup_part'].iloc[i]

        if makeup_part == target_makeup_type and product_id != target_product_id:
            product_name = df_produk['product_name'].iloc[i]
            if product_name not in unique_products:  # Check for uniqueness
                unique_products.add(product_name)  # Add to set
                similar_products_filtered.append({
                    "product_id": int(product_id),
                    "product_name": product_name,
                    "makeup_part": makeup_part,
                    "score": float(score)
                })
        
        if len(similar_products_filtered) >= top_n:
            break
    
    # Limit to top_n products
    limited_unique_products = similar_products_filtered[:top_n]

    return limited_unique_products, target_makeup_type


@app.route('/recommend/cbf/tfidf', methods=['GET'])
def recommend_cbf_tfidf():
    target_product_id = int(request.args.get('product_id'))
    skin_type = request.args.get('skin_type', '')
    skin_tone = request.args.get('skin_tone', '')
    under_tone = request.args.get('under_tone', '')
    top_n = int(request.args.get('top_n', ''))

    top_similar_products, target_makeup_type = cbf_tfidf(target_product_id, skin_type, skin_tone, under_tone, top_n)

    response = {
        "target_makeup_type": target_makeup_type,
        "top_similar_products": [
            {
                "product_id": int(product["product_id"]),  # Konversi ke int
                "product_name": product["product_name"],
                "makeup_part": product.get("makeup_part"),  # Jika ada
                "score": float(product["score"])  # Pastikan ini adalah float
            } for product in top_similar_products[:top_n]
        ]
    }
    
    return jsonify(response)

@app.route('/recommend/cbf/word2vec', methods=['GET'])
def recommend_cbf_word2vec():
    target_product_id = int(request.args.get('product_id'))
    skin_type = request.args.get('skin_type', '')
    skin_tone = request.args.get('skin_tone', '')
    under_tone = request.args.get('under_tone', '')
    top_n = int(request.args.get('top_n', ''))

    top_similar_products, target_makeup_type = cbf_word2vec(target_product_id, skin_type, skin_tone, under_tone, top_n)

    response = {
        "target_makeup_type": target_makeup_type,
        "top_similar_products": [
            {
                "product_id": int(product["product_id"]),  # Konversi ke int
                "product_name": product["product_name"],
                "makeup_part": product.get("makeup_part"),  # Jika ada
                "score": float(product["score"])  # Pastikan ini adalah float
            } for product in top_similar_products[:top_n]
        ]
    }
    
    return jsonify(response)

def svd(user_id, target_product_id, skin_type='', skin_tone='', under_tone='', top_n='', test_size=0.2, n_factors=20, n_epochs=20, lr_all=0.01, reg_all=0.1):
    # Load data from MongoDB collections
    ratings_collection = db['review_product']
    products_collection = db['desc_product_full']

    # Convert MongoDB collections to pandas DataFrames
    
    data = pd.DataFrame(list(ratings_collection.find()))
    products = pd.DataFrame(list(products_collection.find()))

    target_product_row = products[products['product_id'] == target_product_id]
    if target_product_row.empty:
        return jsonify([])

    target_makeup_type = target_product_row['makeup_part'].values[0]
    filter_conditions = []

    if under_tone:
        filter_conditions.append(data['undertone'] == under_tone)
    if skin_type:
        filter_conditions.append(data['skintype'] == skin_type)
    if skin_tone:
        filter_conditions.append(data['skintone'] == skin_tone)

    if filter_conditions:
        filtered_data = data[np.logical_and.reduce(filter_conditions)]

        while len(filtered_data) < 2076 and filter_conditions:
            filter_conditions.pop()
            filtered_data = data[np.logical_and.reduce(filter_conditions)] if filter_conditions else data
    else:
        filtered_data = data

    # Train-test split
    train_data, test_data = train_test_split(filtered_data, test_size=test_size, random_state=42)
    reader = Reader(rating_scale=(1, 5))

    trainset = Dataset.load_from_df(train_data[['user_id', 'product_id', 'stars']], reader).build_full_trainset()
    testset = Dataset.load_from_df(test_data[['user_id', 'product_id', 'stars']], reader).build_full_trainset().build_testset()

    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    model.fit(trainset)

    # Calculate RMSE and MAE
    predictions = model.test(testset)
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)

    all_items = filtered_data['product_id'].unique()
    user_ratings = filtered_data[filtered_data['user_id'] == user_id]
    rated_items = user_ratings['product_id'].unique()
    unrated_items = [item for item in all_items if item not in rated_items]

    predicted_ratings = [(item, model.predict(user_id, item).est) for item in unrated_items]
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    recommendations = [(pred[0], round(pred[1], 4)) for pred in predicted_ratings[:]]
    recommendations_df = pd.DataFrame(recommendations, columns=['product_id', 'predicted_rating'])

    merged_recommendations = recommendations_df.merge(products[['product_id', 'product_name', 'makeup_part']], on='product_id', how='left')

    similar_products_filtered = []
    unique_products = set()

    for _, row in merged_recommendations.iterrows():
        product_id = row['product_id']
        makeup_part = row['makeup_part']
        score = row['predicted_rating']

        if makeup_part == target_makeup_type and product_id != target_product_id:
            product_name = row['product_name']
            if product_name not in unique_products:
                unique_products.add(product_name)
                similar_products_filtered.append({
                    "product_id": int(product_id),
                    "product_name": product_name,
                    "makeup_part": makeup_part,
                    "predicted_rating": float(score)
                })

        if len(similar_products_filtered) >= top_n:
            break

    return similar_products_filtered, target_makeup_type

@app.route('/recommend/svd', methods=['GET'])
def recommend_svd():
    user_id = request.args.get('user_id', type=int)
    target_product_id = request.args.get('product_id', type=int)
    skin_type = request.args.get('skin_type', default='', type=str).lower()
    skin_tone = request.args.get('skin_tone', default='', type=str).lower()
    under_tone = request.args.get('under_tone', default='', type=str).lower()
    top_n = request.args.get('top_n', default='', type=int)  # Default to 10 if not provided

    # Call the recommend_svd function
    top_similar_products, target_makeup_type = svd(user_id, target_product_id, skin_type, skin_tone, under_tone, top_n)

    # Construct the Flask response
    response = {
        "target_makeup_type": target_makeup_type,
        "top_similar_products": [
            {
                "product_id": int(product["product_id"]),
                "product_name": product["product_name"],
                "makeup_part": product["makeup_part"],
                "predicted_rating": float(product["predicted_rating"])
            } for product in top_similar_products[:top_n]  # Limit to top N products
        ]
    }

    # Return JSON response
    return jsonify(response)

def normalize_ratings(df, column_name):
    min_rating = df[column_name].min()
    max_rating = df[column_name].max()

    # Normalisasi menggunakan .loc[] untuk menghindari SettingWithCopyWarning
    df.loc[:, column_name] = (df[column_name] - min_rating) / (max_rating - min_rating)
    return df


def hybrid_recommendations(user_id, target_product_id, skin_type='', skin_tone='', under_tone='', top_n=''):
    ratings_collection = db['review_product']
    products_collection = db['desc_product_full']
    
    data = pd.DataFrame(list(ratings_collection.find()))
    products = pd.DataFrame(list(products_collection.find()))
    
    # Get similar products and target makeup type using CBF
    similar_products, target_makeup_type = cbf_word2vec(target_product_id, skin_type, skin_tone, under_tone, top_n)

    # Ensure similar_products is a DataFrame
    if isinstance(similar_products, list):
        similar_products = pd.DataFrame(similar_products)

    # Get SVD products and target makeup type
    svd_products, target_makeup_type = svd(user_id, target_product_id, skin_type, skin_tone, under_tone, top_n)

    # Ensure svd_products is a DataFrame
    if isinstance(svd_products, list):
        svd_products = pd.DataFrame(svd_products)

    # Normalize ratings
    normalized_df = normalize_ratings(svd_products, 'predicted_rating')

    # Ensure normalized_df is a DataFrame
    if isinstance(normalized_df, list):
        normalized_df = pd.DataFrame(normalized_df)

    all_items = products['product_id'].unique()
    unrated_items = [item for item in all_items if item not in data[data['user_id'] == user_id]['product_id'].unique()]

    # Merge similar products with normalized ratings
    combined_df = pd.merge(similar_products, normalized_df, on='product_id', how='inner')
    combined_df['final_score'] = (0.3 * combined_df['score']) + (0.7 * combined_df['predicted_rating'])

    filtered_combined_df = combined_df[combined_df['product_id'].isin(unrated_items)]

    combined_df_sorted = filtered_combined_df.sort_values(by='final_score', ascending=False)

    similar_products_filtered = []
    unique_products = set()

    for _, row in combined_df_sorted.iterrows():
        product_id = row['product_id']
        makeup_part = row['makeup_part']
        score = row['final_score']

        if makeup_part == target_makeup_type and product_id != target_product_id:
            product_name = row['product_name']
            if product_name not in unique_products:
                unique_products.add(product_name)
                similar_products_filtered.append({
                    "product_id": int(product_id),
                    "product_name": product_name,
                    "makeup_part": makeup_part,
                    "final_score": float(score)
                })

        if len(similar_products_filtered) >= top_n:
            break

    return similar_products_filtered, target_makeup_type

@app.route('/recommend/hybrid', methods=['GET'])
def recommend_hybrid():
    user_id = request.args.get('user_id', type=int)
    target_product_id = request.args.get('product_id', type=int)
    skin_type = request.args.get('skin_type', default='', type=str).lower()
    skin_tone = request.args.get('skin_tone', default='', type=str).lower()
    under_tone = request.args.get('under_tone', default='', type=str).lower()
    top_n = request.args.get('top_n', default='', type=int)  # Default to 10 if not provided

    # Call the hybrid_recommendations function
    top_similar_products, target_makeup_type = hybrid_recommendations(user_id, target_product_id, skin_type, skin_tone, under_tone, top_n)

    # Construct the Flask response
    response = {
        "target_makeup_type": target_makeup_type,
        "top_similar_products": [
            {
                "product_id": int(product["product_id"]),
                "product_name": product["product_name"],
                "makeup_part": product["makeup_part"],
                "predicted_rating": float(product["final_score"])
            } for product in top_similar_products[:top_n]  # Limit to top N products
        ]
    }

    # Return JSON response
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
CORS(app)
