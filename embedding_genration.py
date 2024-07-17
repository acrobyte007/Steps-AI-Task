import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Load the CSV file
df = pd.read_csv('clean_data_2.csv')

# Preprocess the 'information and text' column
df['sentences'] = df['information and text'].apply(sent_tokenize)

from sentence_transformers import SentenceTransformer

# Load BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Calculate embeddings for sentences
df['embeddings'] = df['sentences'].apply(lambda x: model.encode(x, convert_to_tensor=True))
