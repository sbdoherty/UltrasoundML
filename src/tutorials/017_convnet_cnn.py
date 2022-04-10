import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True, as_supervised=True
)

get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
plt.show()

img_size = 160

# need to scale images to all the same size, better to compress than expand
def format_example(image, label):
    """returns an image that is reshaped to IMG_SIZE"""
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (img_size, img_size))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
    plt.figure()
    print(image.numpy())
    plt.imshow(image.numpy())
    plt.title(get_label_name(label))
plt.show()

batch_size = 32
shuffle_buffer_size = 1000
train_batches = train.shuffle(shuffle_buffer_size).batch(batch_size)
validation_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)

img_shape = (img_size, img_size, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
base_model.summary()

for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
print(feature_batch.shape)

#freezing the base convnet (or other models)
base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])
model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
print(history.history['accuracy'])

model.save('dogs_vs_cats.h5')
new_model = tf.keras.models.load_model('dogs_vs_cats.h5') #facial recognition module in python?


