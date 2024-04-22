from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question, emotions, prompt):
    emotion_text = " ".join([f"{feeling}:{weight}" for feeling, weight in emotions.items()])
    input_with_emotion = f"{question} [{emotion_text}] {prompt}"
    response = chat.send_message(input_with_emotion, stream=True)
    return response

st.set_page_config(page_title="MITRAI-CHATBOT")

st.header("MITRAI")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input: ", key="input")
submit = st.button("Hi!How do you do?")

prompt = """
You are an expert mental therapist. User will start conversation with you and
we'll give you the emotions in the form of an array having 7 emotions joy, disgust, fear, sad, surprise, neutral, angry, the array will have weight value of each emotion between 0 and 1, sum of all weights will be 1
and you will have to give the user therapeutic advice with the help of your expertise keeping in mind the emotions with significant weights.
Keep your responses short, inquiring more about the user's situation and try too resolve it or support it, where applicable
"""

# Load the emotion detection pipeline from the Hugging Face model hub
from transformers import pipeline

emotion_detection_pipeline = pipeline("text-classification", model="badmatr11x/roberta-base-emotions-detection-from-text", tokenizer="badmatr11x/roberta-base-emotions-detection-from-text")

if input:
    user_text = input
    emotions = emotion_detection_pipeline(user_text, top_k=None)

    predicted_emotion = {}
    for emotion in emotions:
        label = emotion['label']
        score = emotion['score']
        predicted_emotion[label] = score
    print(predicted_emotion)
    response = get_gemini_response(input, predicted_emotion, prompt)
    st.session_state['chat_history'].append(("You", input))
    st.subheader("MitrAI: ")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("MitrAI", chunk.text))

show_history = st.checkbox("Show/Hide History")
if show_history:
    st.subheader("History:")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")
