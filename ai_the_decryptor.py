import base64
import binascii
import codecs
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys  

# Types of encodings
encodings = ["HEX", "Base64", "ROT13", "URL-Encoding"]

# The decoder
def decode_string(encoded_str, encoding_type):
    try:
        if encoding_type == "HEX":
            return binascii.unhexlify(encoded_str).decode('utf-8')
        elif encoding_type == "Base64":
            return base64.b64decode(encoded_str).decode('utf-8')
        elif encoding_type == "ROT13":
            return codecs.decode(encoded_str, 'rot_13')
        elif encoding_type == "URL-Encoding":
            return codecs.decode(encoded_str.replace('%', '\\x'), 'unicode_escape')
    except Exception:
        return None 

# Training data
data = [
    ("48656c6c6f20576f726c64", 0),     # HEX
    ("SGVsbG8gV29ybGQ=", 1),          # Base64
    ("Uryyb Jbeyq", 2),               # ROT13
    ("%48%65%6c%6c%6f", 3)            # URL-Encoding
]

X_train = [list(map(ord, item[0])) for item in data]
y_train = [item[1] for item in data]

X_train = keras.preprocessing.sequence.pad_sequences(X_train, padding='post')
y_train = np.array(y_train)

# The Module
model = keras.Sequential([
    keras.layers.Embedding(input_dim=256, output_dim=16, input_length=X_train.shape[1]),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(encodings), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, verbose=0)

# The Predict
def predict_encoding(encoded_str):
    input_data = list(map(ord, encoded_str))
    input_data = keras.preprocessing.sequence.pad_sequences([input_data], maxlen=X_train.shape[1], padding='post')
    prediction = model.predict(input_data, verbose=0)
    encoding_index = np.argmax(prediction)
    return encodings[encoding_index]

# Launch
if __name__ == "__main__":
    print("🧠 The decryptor is active.")
    print("Enter the encrypted string. Enter 'exit' to exit manually..")

    while True:
        encoded = input("\n🔐 Enter the line: ")
        if encoded.strip().lower() == "exit":
            print("👋 Exit manually.")
            break

        encoding = predict_encoding(encoded)
        decoded = decode_string(encoded, encoding)

        if decoded is None:
            print(f"⛔ Error: cannot be decrypted with the format {encoding}.")
            print("🔁 Return to the main menu...")
            break
        else:
            print(f"🔎 Format: {encoding}")
            print(f"📥 Decoding: {decoded}")
