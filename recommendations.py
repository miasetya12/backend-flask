import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
from db import get_collection  # Import fungsi untuk mendapatkan koleksi

# def cbf_tfidf(target_product_id, skin_type='', skin_tone='', under_tone='', top_n=''):
#     collection = get_collection()  # Mendapatkan koleksi
#     df_produk = pd.DataFrame(list(collection.find()))
#     df_produk['unique_data_clean'] = df_produk['unique_data_clean'].astype(str).fillna('')
#     target_product_row = df_produk[df_produk['product_id'] == target_product_id]
#     if target_product_row.empty:
#         return []
#     target_product_description = target_product_row['unique_data_clean'].values[0]
#     target_makeup_type = target_product_row['makeup_part'].values[0]

#     if skin_type:
#         target_product_description += f" {skin_type}"
#     if skin_tone:
#         target_product_description += f" {skin_tone}"
#     if under_tone:
#         target_product_description += f" {under_tone}"

#     df_produk.loc[target_product_row.index, 'unique_data_clean'] = target_product_description
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(df_produk['unique_data_clean'])
#     target_vector = tfidf_matrix[df_produk.index[df_produk['product_id'] == target_product_id][0]]
#     cosine_sim = cosine_similarity(target_vector, tfidf_matrix)

#     similarity_scores = list(enumerate(cosine_sim[0]))
#     sorted_similar_items = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
#     similar_products_filtered = []
#     for i, score in sorted_similar_items:
#         product_id = df_produk['product_id'].iloc[i]
#         makeup_part = df_produk['makeup_part'].iloc[i]
#         if makeup_part == target_makeup_type and product_id != target_product_id:
#             product_name = df_produk['product_name'].iloc[i]
#             similar_products_filtered.append({
#                 "product_id": int(product_id),
#                 "product_name": product_name,
#                 "makeup_part": makeup_part,
#                 "score": float(score)
#             })
#         if len(similar_products_filtered) >= top_n * 3:
#             break
    
#     unique_products = {}
#     for product_id, product_name, makeup_part, score in similar_products_filtered:
#         if product_name not in unique_products:
#             unique_products[product_name] = (product_id, makeup_part, score)

#     # Limit the number of unique products based on unique_count
#     limited_unique_products = dict(list(unique_products.items())[:top_n])

#     # unique_products = {}
    
#     # for i, score in sorted_similar_items:
#     #     product_id = df_produk['product_id'].iloc[i]
#     #     makeup_part = df_produk['makeup_part'].iloc[i]
#     #     product_name = df_produk['product_name'].iloc[i]

#     #     # Pastikan produk adalah unik berdasarkan product_name
#     #     if product_name not in unique_products and product_id != target_product_id:
#     #         unique_products[product_name] = {
#     #             "product_id": int(product_id),
#     #             "product_name": product_name,
#     #             "makeup_part": makeup_part,
#     #             "score": float(score)
#     #         }

#     #     # Batasi jumlah produk unik berdasarkan top_n
#     #     if len(unique_products) >= top_n:
#     #         break

#     # return list(unique_products.values()), target_makeup_type
#     return limited_unique_products, target_makeup_type
#     # return similar_products_filtered, target_makeup_type


def cbf_tfidf(target_product_id, skin_type='', skin_tone='', under_tone='', top_n=''):
    collection = get_collection()  # Get collection
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
    collection = get_collection()  # Get collection
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
