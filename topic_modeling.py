from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

# Flatten the list of sentences for LDA
all_sentences = [sentence for sublist in df['sentences'].tolist() for sentence in sublist]
dictionary = Dictionary([nltk.word_tokenize(text) for text in all_sentences])
corpus = [dictionary.doc2bow(nltk.word_tokenize(text)) for text in all_sentences]
lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary)
