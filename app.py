import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Load custom trained model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('https://huggingface.co/spaces/SoniR/chatbotllm/blob/main/config.json')
# tokenizer = GPT2Tokenizer.from_pretrained('https://huggingface.co/spaces/SoniR/chatbotllm/blob/main/pytorch_model.bin')

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="gpt2")# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Streamlit app title
st.title("Custom Trained Chatbot")

# Function to generate chatbot response
def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main UI components
user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    bot_response = generate_response(user_input)
    st.write("Bot:", bot_response)
