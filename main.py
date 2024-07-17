import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Load the CSV file
df = pd.read_csv('clean_data_2.csv')

# Preprocess the 'information and text' column
df['sentences'] = df['information and text'].apply(sent_tokenize)

from sentence_transformers import SentenceTransformer, util
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

# Load BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Calculate embeddings for sentences
df['embeddings'] = df['sentences'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# LDA for topic modeling
# Flatten the list of sentences for LDA
all_sentences = [sentence for sublist in df['sentences'].tolist() for sentence in sublist]
dictionary = Dictionary([nltk.word_tokenize(text) for text in all_sentences])
corpus = [dictionary.doc2bow(nltk.word_tokenize(text)) for text in all_sentences]
lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary)




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



from sentence_transformers import CrossEncoder

# Define BM25 and BERT-based models
bm25_model = ...
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Query the collection
query_vector = model.encode(query_text, convert_to_tensor=True)
search_params = {"metric_type": "IP", "params": {"ef": 128}}
results = collection.search(query_vector, "embedding", search_params, limit=10)

# Re-rank using CrossEncoder
reranked_results = cross_encoder.predict([(query_text, result["text"]) for result in results])




from transformers import pipeline

# Load question answering pipeline
qa_pipeline = pipeline("question-answering")

# Generate answer
answer = qa_pipeline(question=query_text, context=reranked_results[0]["text"])
print(answer)



