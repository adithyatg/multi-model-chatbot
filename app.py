import os
import streamlit as st
from transformers import pipeline

# Title
st.title("Model Selection Chatbot")

# Prompt input
prompt = st.text_input("Enter your prompt:")

# Model selection
model_choice = st.selectbox("Choose a model:", ["GPT", "BERT", "BART", "Llama", "Mistral"])

# Hugging Face API token
huggingface_token = os.getenv("HF_TOKEN", default="")

# Dictionary to load models
models = {
    "GPT": "openai-community/openai-gpt",
    "BERT": "deepset/bert-base-cased-squad2",
    "BART": "facebook/bart-large",
    "Llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral": "mistralai/Mistral-7B-v0.1" 
}

# Submit button
if st.button("Submit"):
    try:
        # Load the selected model
        if model_choice == "BERT":
            model = pipeline('question-answering', model=models[model_choice], use_auth_token=huggingface_token)
            response = model(question=prompt, context=prompt)
            st.write(response['answer'])
        elif model_choice == "BART":
            model = pipeline('summarization', model=models[model_choice], use_auth_token=huggingface_token)
            response = model(prompt)
            st.write(response[0]['summary_text'])
        elif model_choice == "GPT":
            model = pipeline('text-generation', model=models[model_choice], use_auth_token=huggingface_token)
            response = model(prompt)
            st.write(response[0]['generated_text'])
        elif model_choice == "Llama":
            model = pipeline('text-generation', model=models[model_choice], use_auth_token=huggingface_token)
            response = model(prompt)
            st.write(response[0]['generated_text'])
        elif model_choice == "Mistral":
            model = pipeline('text-generation', model=models[model_choice], use_auth_token=huggingface_token)
            response = model(prompt)
            st.write(response[0]['generated_text'])
    except Exception as e:
        st.error(f"An error occurred: {e}")
