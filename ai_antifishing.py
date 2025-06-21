import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys  

# Learning function
def train_bot_scam_detector(texts, labels, epochs=100):
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=50, padding='post')

    model = keras.Sequential([
        keras.layers.Embedding(input_dim=5000, output_dim=16, input_length=50),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    class_weight = {0: 1.0, 1: 0.8}
    model.fit(np.array(padded_sequences), np.array(labels), epochs=epochs, batch_size=8, validation_split=0.2, class_weight=class_weight)

    return model, tokenizer

# Here basic texts for AI
texts = [
    "How are you?", "I'm at home, waiting for you.", "Go to the store and buy bread.",
    "Shall we go for a walk in the evening?", "The weather is good today, let's take a walk!",
    "I'm working today, see you later.", "How was your day?",
    "I'm going to watch a movie, are you with me?", "Let's meet at the cafe tomorrow!",
    "It's been a long day, I'm tired.",
    "Your account is blocked, contact support to unlock it.",
    "Our company has chosen you to participate in an exclusive promotion!",
    "Your bank notifies you of the urgent need to update your card details.",
    "Hello, the support service is bothering you, please send your details.",
    "Congratulations, you have won the prize! Send us your banking details.",
    "Your account will be blocked if you do not verify your details!",
    "Pay the debt urgently, otherwise you will be called to court!",
    "Special offer! Invest $1,000 and get a bonus!",
    "Your PayPal account is blocked. Please follow the recovery link urgently!",
    "You've won a car! Pay the tax to receive the prize.",
    "Your mobile operator gives you 10 GB of Internet! Just send the code!",
    "Verify your bank account, otherwise it will be deleted."
]

labels = [0]*10 + [1]*12  # 0 - for Human, 1 - for Bot

# Training
model, tokenizer = train_bot_scam_detector(texts, labels)

# Predict
def predict_text(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post')
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    return "Fraudster/bot" if prediction > 0.5 else "Human"

# Start
if __name__ == "__main__":
    print("\n🧠 The anti-phishing AI is active. Enter the text for analysis.")
    print("Enter 'exit' to exit manually.")

    while True:
        text = input("\nText > ")
        if text.lower().strip() == "exit":
            print("👋 Manual completion.")
            break

        result = predict_text(text, model, tokenizer)
        print(f"The Result: {result}")

        if result == "Fraudster/bot":
            print("⛔ A phishing or bot text has been detected! Return to the main menu...")
            break
        else:
            print("✅ Secure text. Continue typing.")
