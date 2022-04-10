from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file("shakespeare.txt",
                                       "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

#Read in the file
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
print(f"Length of text is {len(text)}")

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])


def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

text_as_int = text_to_int(text)
print(text_as_int)
print(int_to_text(text_as_int[50:100]))

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1) #floor division
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

#making training batches
batch_size = 64
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
buffer_size = 10000
data = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

