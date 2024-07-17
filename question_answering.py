from transformers import pipeline
import question_answering
import embeddings

# Initialize question answering pipeline
qa_pipeline = question_answering.initialize_qa_pipeline()

# Interactive question answering loop
while True:
    # Prompt user for question input
    query_text = input("Enter your question (or 'exit' to quit): ")
    
    # Check if user wants to exit
    if query_text.lower() == 'exit':
        break
    
    # Perform question answering
    query_vector = embeddings.calculate_embeddings([query_text])[0]
    search_params = {"metric_type": "IP", "params": {"ef": 128}}
    results = collection.search(query_vector, "embedding", search_params, limit=10)
    reranked_results = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').predict([(query_text, result["text"]) for result in results])
    answer = question_answering.perform_qa(query_text, reranked_results[0]["text"], qa_pipeline)
    
    # Display the answer
    print(f"Answer: {answer['answer']}")

