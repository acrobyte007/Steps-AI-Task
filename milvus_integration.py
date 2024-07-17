from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Change dim accordingly
    FieldSchema(name="metadata", dtype=DataType.JSON)
]
schema = CollectionSchema(fields, "semantic_search")

# Create collection
collection = Collection("text_chunks", schema)

# Insert data
for idx, row in df.iterrows():
    entities = [
        [idx],  # ID
        row['embeddings'].tolist(),  # Embedding vector
        {"link": row["link"], "topic_name": row["topic name"]}  # Metadata
    ]
    collection.insert(entities)

# Create HNSW index
index_params = {"index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
collection.create_index(field_name="embedding", index_params=index_params)
