from fastprogress.fastprogress import MasterBar, ProgressBar, master_bar, progress_bar


def noop(*args, **kwargs):
    pass

master_bar.update = noop
master_bar.on_update = noop
master_bar.show = noop
progress_bar.update = noop
progress_bar.on_update = noop
progress_bar.show = noop


class DummyBar:
    def __init__(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def on_iter_begin(self, *args, **kwargs): pass
    def on_iter_end(self, *args, **kwargs): pass
    def on_update(self, *args, **kwargs): pass
    def on_interrupt(self, *args, **kwargs): pass

MasterBar = ProgressBar = DummyBar


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

import logging
logging.getLogger("ktrain").setLevel(logging.WARNING)

from openai import AzureOpenAI
from transformers import pipeline
from ktrain import text
import shutil
import time
import re



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
    # Use a unique directory for the index
    timestamp = int(time.time())
    index_dir = f'./tmp/myindex_{timestamp}'
    docs_folder = './doc/'

    # Ensure the directory does not already exist
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)


    # Initialize the index (creates the directory automatically)
    text.SimpleQA.initialize_index(index_dir)

    # Index documents
    text.SimpleQA.index_from_folder(
        docs_folder,
        index_dir=index_dir,
        use_text_extraction=True,
        commit_every=1
    )

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
        # Add context as the system message
        context_message = {
            "role": "system",
            "content": (
                "You are a multilingual expert waiter assisting customers with menu inquiries. "
                "Answer questions about the menu in an energetic and respectful tone. "
                "Provide information only from the data. If you cannot answer, respond with: "
                '"Sorry, I cannot answer that question. Please ask the waiter or call 9001-0304."\n\n'
                "Example:\n\n"
                "Question: What are the best gluten-free spicy options?\n"
                "Answer:\n"
                "[Dishes]: The best gluten-free and spicy dishes are Cacio and Tartufo $10 and Duomo di Parma $15.\n"
                "[Details]: The first is served directly on top of a Pecorino cheese round, and the second features "
                "Jamón de Parma, San Marzano tomatoes, stracciatella, fresh basil, olives, and a touch of paprika.\n"
                "Output Indicators: Use [Dishes] for the list of dishes with [price] and [Details] for further descriptions. "
                "If unable to answer, use [Unavailable] with the provided fallback message."
            )
        }

        # Append the context message to the conversation history
        conversation_history.insert(0, context_message)

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
        # Clean the response
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
