from flask import jsonify, request
from bson.json_util import dumps
from db import get_collection
from recommendations import cbf_tfidf, cbf_word2vec

def configure_routes(app):
    collection = get_collection()

    @app.route('/products', methods=['GET'])
    def get_products():
        """Fetch all product descriptions from MongoDB."""
        try:
            products = collection.find().sort("price", -1).limit(20)
            return dumps(products)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    @app.route('/products/<int:product_id>', methods=['GET'])
    def get_product(product_id):
        """Fetch a product description by ID from MongoDB."""
        try:
            product = collection.find_one({"product_id": product_id})
            if product:
                return dumps(product)
            else:
                return jsonify({"error": "Product not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/recommend/cbf/tdidf', methods=['GET'])
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

