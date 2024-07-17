Requirements
Python 
Libraries:
pandas
nltk
sentence_transformers
gensim
pymilvus
transformers (from Hugging Face)
Installation
Clone the repository:https://github.com/acrobyte007/Steps-AI-Task

bash
Copy code
git clone 
cd your-repository
Install dependencies:

Copy code
pip install -r requirements.txt
Usage
Data Preprocessing:

Ensure your data (clean_data_2.csv) is prepared and accessible.
Embedding Generation:

Run embedding_generation.py to preprocess text data and generate sentence embeddings using BERT.
Topic Modeling (LDA):

Run topic_modeling.py to perform LDA topic modeling on the preprocessed text.
MILVUS Integration:

Start MILVUS server locally (localhost:19530).
Run milvus_integration.py to insert embeddings into MILVUS and create an HNSW index.
Querying and Re-ranking:

Adjust parameters in query_and_rerank.py for query execution and result re-ranking.
Question Answering:

Run question_answering.py to perform question answering using the retrieved and reranked results.
Files and Structure
embedding_generation.py: Script for generating embeddings from text data.
topic_modeling.py: Script for LDA topic modeling.
milvus_integration.py: Script for integrating with MILVUS for embedding storage and retrieval.
query_and_rerank.py: Script for querying MILVUS and re-ranking results.
question_answering.py: Script for question answering using the retrieved and reranked results.
