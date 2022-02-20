import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def fashion_classification():
    """Simple NN about image classification based on TF tutorial. """
    fashion_mnist = keras.datasets.fashion_mnist  # 60000 images for training, 10000 images for testing
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Brief look at the dataset
    print(train_images.shape)  # 60000 images, at 28x28 pixels
    print(train_images[0, 23, 23]) # 194, what one pixel looks like, 0-255 for grayscale values
    print(train_labels[:10])

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    # Preprocess to normalize between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer, turn image into a flat vector of 784 neurons
        keras.layers.Dense(128, activation='relu'),  # hidden layer, usually will be smaller than input layer
        keras.layers.Dense(10, activation='softmax')  #output layer, softmax creates a pdf of image for the 10 classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )  # this can be changed, as hyperparameter tuning. How does model perform if we change values?

    model.fit(train_images, train_labels, epochs=1)  # will more epochs overfit the data?

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

    print('Test accuracy: {}'.format(test_acc))

    predictions = model.predict(test_images) # put a single item as an array if needed
    #print(class_names[np.argmax(predictions[0])])

    # # Plotting of an image
    # plt.figure()
    # plt.imshow(train_images[0], cmap='gray_r', vmin=0, vmax=1)
    # plt.colorbar()
    # # plt.grid(True)
    # plt.show()

    visualize_data = True
    while visualize_data is True:
        num = get_number()
        image = test_images[num]
        label = test_labels[num]
        predict(model, image, label, class_names)
        loop_bool = input("Continue making predictions? Input 'T' or 'F': ")
        if loop_bool == 'T':
            pass
        else:
            visualize_data = False


def predict(model, image, correct_label, class_names):
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
    """Display an image. Show the ML guess as well as the actual label. """
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("ML Prediction: " + guess)
    plt.colorbar()
    #plt.grid = True
    plt.show()


def get_number():
    """Receive input and verify value is an integer. """
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
            else:
                print("Only numbers accepted!")

if __name__ == '__main__':
    fashion_classification()


