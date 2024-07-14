import streamlit as st
from transformers import pipeline

st.title("Chatbot")

prompt = st.text_input("Enter your prompt:")

model_choice = st.selectbox("Choose a model:", ["gpt-3.5-turbo", "bert", "bart", "llama", "mistral"])

# Dictionary to load models
models = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "bert": "bert-base-uncased",
    "bart": "facebook/bart-large-cnn",
    "llama": "facebook/llama",
    "mistral": "mistral-ai/mistral-7b"
}

# Submit button
if st.button("Submit"):
    # Load the selected model
    if model_choice == "bert":
        model = pipeline('fill-mask', model=models[model_choice])
    elif model_choice == "bart":
        model = pipeline('summarization', model=models[model_choice])
    else:
        model = pipeline('text-generation', model=models[model_choice])
        
    # Generate response
    if model_choice == "bert":
        response = model(f"{prompt} [MASK].")
        st.write(response[0]['sequence'])
    elif model_choice == "bart":
        response = model(prompt)
        st.write(response[0]['summary_text'])
    else:
        response = model(prompt)
        st.write(response[0]['generated_text'])
