import streamlit as st
from haystack.telemetry import tutorial_running
from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
import os

# Load API key from Streamlit Secrets or environment variable
if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.environ["OPENAI_API_KEY"]
else:
    openai_api_key = st.secrets["openai_api_key"]

# Initialize document store
document_store = InMemoryDocumentStore()

# Load dataset from uploaded CSV using datasets
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    dataset = load_dataset('csv', data_files={'train': uploaded_file})
    dataset = dataset['train'].map(lambda example: {'title': example['title'], 'abstract': example['abstract']})
    docs = [Document(content=example['abstract'], meta={'title': example['title']}) for example in dataset]
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

# Initialize SentenceTransformers document embedder
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

# Initialize SentenceTransformers text embedder
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Initialize retriever
retriever = InMemoryEmbeddingRetriever(document_store)

# Initialize prompt builder
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=template)

# Initialize OpenAIGenerator
generator = OpenAIGenerator(model="gpt-3.5-turbo")

# Set API key if available
if openai_api_key:
    generator.client.set_api_key(openai_api_key)

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
