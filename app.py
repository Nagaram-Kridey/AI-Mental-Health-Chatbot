from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load trained model
model = tf.keras.models.load_model("mental_health_chatbot.keras")

# Tokenizer (same as training)
VOCAB_SIZE = 5000
MAX_LEN = 20
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters="")

app = FastAPI()

def chatbot_response(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
    prediction = model.predict([padded, padded])
    response_seq = np.argmax(prediction[0], axis=1)
    response = " ".join([tokenizer.index_word[idx] for idx in response_seq if idx > 0])
    return response

@app.get("/chat")
def chat(user_input: str):
    response = chatbot_response(user_input)
    return {"bot_response": response}
