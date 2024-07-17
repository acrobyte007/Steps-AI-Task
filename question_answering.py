from transformers import pipeline

# Load question answering pipeline
qa_pipeline = pipeline("question-answering")

# Generate answer
answer = qa_pipeline(question=query_text, context=results[0]["text"])
print(answer)
