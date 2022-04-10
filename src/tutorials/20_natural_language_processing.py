"""Recurrent Neural Network for NLP (sentiment analysis on movie reviews)"""

from keras.datasets import imdb
from keras.preprocessing import sequence
import os
import numpy as np
import tensorflow as tf

vocab_size = 88584 #sorted by most common, ie integer 1 is most common word

max_len = 250
batch_size = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

print(train_data[0])

train_data = sequence.pad_sequences(train_data, max_len)
test_data = sequence.pad_sequences(test_data, max_len)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.summary()
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], max_len)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

reverse_word_index = {value: key for (key, value) in word_index.items()}
def decode_integers_to_text(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]


print(decode_integers_to_text(encoded))


def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

positive_review = "That movie was totally awesome! I was engaged the whole film and can't wait for the sequel to come out."
negative_review = "That movie was horrible, I hated the actors and the plot was difficult to follow. I would not recommend to a friend."

predict(positive_review)
predict(negative_review)