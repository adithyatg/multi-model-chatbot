import streamlit as st
import os
import tempfile
from haystack.telemetry import tutorial_running
from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline

# Check tutorial running
tutorial_running(27)

# Title and file upload
st.title("Haystack Pipeline Demo")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# Input for OpenAI API key
openai_api_key = "sk-proj-RFxwtsKgU22C66rt7ZSlT3BlbkFJMTErq0GLFGSGvfCOezWw"

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load dataset from uploaded CSV using datasets
    dataset = load_dataset('csv', data_files={'train': temp_file_path})
    dataset = dataset['train'].map(lambda example: {'title': example['title'], 'abstract': example['abstract']})

    # Convert dataset into Haystack Documents
    docs = [Document(content=example['abstract'], meta={'title': example['title']}) for example in dataset]

    # Initialize InMemoryDocumentStore
    document_store = InMemoryDocumentStore()

    # Initialize SentenceTransformersDocumentEmbedder
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()

    # Generate embeddings and write to document store
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

    # Initialize SentenceTransformersTextEmbedder
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize InMemoryEmbeddingRetriever
    retriever = InMemoryEmbeddingRetriever(document_store)

    # Define PromptBuilder template
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    # Initialize PromptBuilder
    prompt_builder = PromptBuilder(template=template)

    # Initialize OpenAIGenerator with static API key
    generator = OpenAIGenerator(model="gpt-3.5-turbo", api_key=openai_api_key)

    # Initialize Pipeline
    basic_rag_pipeline = Pipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Connect components in the pipeline
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    # Input question from user
    question = st.text_area("Enter your question here:")

    # Run pipeline when Submit button is clicked
    if st.button("Submit"):
        try:
            response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
            st.write(response["llm"]["replies"][0])
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Clean up: Remove the temporary file
    os.remove(temp_file_path)

else:
    st.info("Please upload a CSV file to begin.")
