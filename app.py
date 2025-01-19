from openai import AzureOpenAI
from transformers import pipeline
import os
from ktrain import text
import re
import shutil




ENDPOINT = "https://GENAISUSANA.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-4o-mini-2"
SEARCH_ENDPOINT = "https://tryonerag.search.windows.net/"
SEARCH_KEY = "1GWsfiQACTI2TDUMHbwjeWqXVyOT6aIqfAzMUgTVPaAzSeB80uET"
SUBSCRIPTION_KEY = "A72bPWAi8EzXZOXJONzxbayRzSBZQO60oGiJQZtTZfHKEijvNGyYJQQJ99ALAC5RqLJXJ3w3AAABACOGWpqP"

def load_azure_openai_client():
    """Load Azure OpenAI client."""
    return AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
        api_version="2024-05-01-preview",
    )

def load_classifier():
    """Load the zero-shot classification model."""
    return pipeline("zero-shot-classification", model="microsoft/deberta-large-mnli")

def index_documents():
    """Index documents for SimpleQA."""
    index_dir = './tmp/myindex'
    docs_folder = './doc/'

    # Check if the directory exists and remove it if necessary
    if os.path.exists(index_dir):
        print(f"Clearing existing index directory: {index_dir}")
        shutil.rmtree(index_dir)  # Delete the directory and its contents

    # Recreate the directory
    os.makedirs(index_dir, exist_ok=True)

    # Initialize the index
    text.SimpleQA.initialize_index(index_dir)

    # Index documents from the specified folder
    text.SimpleQA.index_from_folder(
        docs_folder,
        index_dir=index_dir,
        use_text_extraction=True,  
        commit_every=1             
    )

    print(f"Documents indexed successfully in: {index_dir}")
    return index_dir



def load_qa_model():
    """Initialize and load the SimpleQA model."""
    index_dir = index_documents()  
    return text.SimpleQA(index_dir)

def classify_query(classifier, user_text):
    """Classify the query into categories."""
    labels = ['Menu Related', 'Not Menu Related', 'Operational Related']
    contextualized_text = f"Classify this question as related to menu, operations, or other: '{user_text}'"
    result = classifier(contextualized_text, labels)
    predicted_label = result['labels'][0]
    return predicted_label

def generate_response(client, conversation_history):
    """Generate response using Azure OpenAI."""
    try:
        completion = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=conversation_history,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            extra_body={
                "data_sources": [{
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": SEARCH_ENDPOINT,
                        "index_name": "qamenufinal",
                        "semantic_configuration": "azureml-default",
                        "authentication": {
                            "type": "api_key",
                            "key": SEARCH_KEY
                        },
                        "embedding_dependency": {
                            "type": "endpoint",
                            "endpoint": f"{ENDPOINT}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-07-01-preview",
                            "authentication": {
                                "type": "api_key",
                                "key": SUBSCRIPTION_KEY
                            }
                        },
                        "query_type": "vector_simple_hybrid",
                        "top_n_documents": 5
                    }
                }]
            }
        )
        response_content = completion.choices[0].message.content.strip()
        response_content = re.sub(r'\[doc\d+\]', '', response_content)
        return response_content
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def ask_question(classifier, qa_model, client, question):
    """Route the query based on classification."""
    classification = classify_query(classifier, question)
    if classification == "Menu Related":
        conversation_history = [{"role": "user", "content": question}]
        return generate_response(client, conversation_history), "azure"
    else:
        # Get answers from the QA model
        answers = qa_model.ask(question)
        if answers:
            full_answer = answers[0]['full_answer']
            extracted_answer = full_answer.split('a :', 1)[-1].strip().lower()
            return extracted_answer, "bert"
        else:
            return "I'm sorry, I couldn't find an answer to your question.", "bert"

def main():
    """Main function."""
    print("Loading models...")
    client = load_azure_openai_client()
    classifier = load_classifier()
    qa_model = load_qa_model()
    print("Models loaded. Ready to answer your questions.")

    while True:
        question = input("\nAsk your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        
        # Generate response
        answer, source = ask_question(classifier, qa_model, client, question)
        print(f"\nAnswer ({source}): {answer}")

if __name__ == "__main__":
    main()
