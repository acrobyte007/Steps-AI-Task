from pymilvus import connections, Collection

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Query the collection
query_vector = model.encode(query_text, convert_to_tensor=True)
search_params = {"metric_type": "IP", "params": {"ef": 128}}
results = collection.search(query_vector, "embedding", search_params, limit=10)
