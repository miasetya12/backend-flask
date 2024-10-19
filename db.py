from pymongo import MongoClient

# Koneksi ke MongoDB Atlas
client = MongoClient("mongodb+srv://miasetyautami:20xO81m3RtBrqBZr@cluster1.ohirn.mongodb.net/")
db = client['makeup_product']
collection = db['desc_product_full']

def get_collection():
    """Return the MongoDB collection."""
    return collection
