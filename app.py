from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson.json_util import dumps
from bson.objectid import ObjectId  # Import ObjectId to handle MongoDB Object IDs



from flask_cors import CORS # type: ignore

app = Flask(__name__)

# Koneksi ke MongoDB Atlas
client = MongoClient("mongodb+srv://miasetyautami:20xO81m3RtBrqBZr@cluster1.ohirn.mongodb.net/")
db = client['makeup_product']
collection = db['desc_product']
collection_full = db['desc_product_full']


# Fungsi untuk mendapatkan produk serupa
def get_top_similar_products(data, target_product_id, skin_type='', makeup_type='', top_n=5):
    df_produk = pd.DataFrame(data)
    df_produk['unique_data_clean'] = df_produk['unique_data_clean'].astype(str).fillna('')

    # Cari baris produk target berdasarkan product_id
    target_product_row = df_produk[df_produk['product_id'] == target_product_id]
    if target_product_row.empty:
        return []
    target_product_description = target_product_row['unique_data_clean'].values[0]
    target_makeup_type = target_product_row['makeup_part'].values[0]

    # Tambahkan skin_type dan makeup_type jika ada
    if skin_type:
        target_product_description += f" {skin_type}"
    if makeup_type:
        target_product_description += f" {makeup_type}"

    # Update deskripsi produk target
    df_produk.loc[target_product_row.index, 'unique_data_clean'] = target_product_description

    # Vectorizer TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_produk['unique_data_clean'])

    # Hitung kemiripan kosinus
    target_vector = tfidf_matrix[df_produk.index[df_produk['product_id'] == target_product_id][0]]
    cosine_sim = cosine_similarity(target_vector, tfidf_matrix)

    # Ambil skor kemiripan
    similarity_scores = list(enumerate(cosine_sim[0]))

    # Urutkan berdasarkan skor tertinggi
    sorted_similar_items = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Pilih top_n produk yang mirip
    top_similar_items = sorted_similar_items[1:top_n + 1]

    similar_products = []
    for i, score in top_similar_items:
        product_name = df_produk['product_name'].iloc[i]
        product_id = int(df_produk['product_id'].iloc[i])  # Mengonversi ke int
        similar_products.append((product_id, product_name, score))

    return similar_products, target_makeup_type


@app.route('/products', methods=['GET'])
def get_products():
    """Fetch all product descriptions from MongoDB."""
    try:
        products = collection_full.find().sort("price", -1).limit(20)  # Sort by price in ascending order
        return dumps(products)  # Use dumps for JSON serialization
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint Flask untuk rekomendasi produk
@app.route('/recommend', methods=['GET'])
def recommend_products():
    try:
        # Ambil parameter dari request
        target_product_id = int(request.args.get('product_id'))
        skin_type = request.args.get('skin_type', '')
        makeup_type = request.args.get('makeup_type', '')
        top_n = int(request.args.get('top_n', 5))

        # Ambil data dari MongoDB
        data = list(collection.find({}, {"_id": 0}))  # Menghilangkan _id dari hasil

        # Dapatkan produk yang mirip
        top_similar_products, target_makeup_type = get_top_similar_products(data, target_product_id, skin_type, makeup_type, top_n)

        # Format hasil menjadi JSON
        result = {
            "target_product_id": target_product_id,
            "target_makeup_type": target_makeup_type,
            "similar_products": [
                {"product_id": int(product_id), "product_name": product_name, "similarity_score": float(score)}  # Mengonversi ke int dan float
                for product_id, product_name, score in top_similar_products
            ]
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/products/<int:product_id>', methods=['GET'])  # Mengubah tipe parameter ke int
def get_product(product_id):
    """Fetch a product description by ID from MongoDB."""
    try:
        print(f"Received product_id: {product_id}")  # Debugging line
        product = collection_full.find_one({"product_id": product_id})  # Query menggunakan product_id sebagai Int
        if product:
            print(product) 
            return dumps(product)  # Serialize the product data
        else:
            return jsonify({"error": "Product not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
CORS(app)