import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import seaborn as sns
from sklearn.model_selection import train_test_split


def plot_loss(head_tail, history):
    """Visualize error decrease over training process """
    plt.figure(2)
    plt.plot(history.history['mape'], label='loss')
    plt.plot(history.history['val_mape'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    if not os.path.exists(os.path.join(head_tail[0], "..", "Pictures")):
        os.mkdir(os.path.join(head_tail[0], "..", "Pictures"))
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", "loss_history.png"))


def build_and_compile_model(norm):
    """Defines the input function to build a deep neural network for tensor flow"""
    model = keras.Sequential([
      norm,
      layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
      layers.Dropout(0.5),
      layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
      layers.Dropout(0.5),
      layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.0001),
                  metrics=["mae", "mse", "mape"])
    return model


def tf_demographics(csv, categorical_features, numerical_features=""):
    # Read in the master_csv file
    dataset = pd.read_csv(csv)
    features = [*categorical_features, *numerical_features]
    # Filter the csv by the desired features
    dataset = dataset[features]
    dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    dataset = dataset.dropna()
    print(dataset.head())  # Show first five entries and some columns
    print(dataset.shape)
    print(dataset.dtypes)

    # Plotting of numeric variables, not necessary for DNN
    try:
        plt.figure(1)
        sns.pairplot(dataset[numerical_features], diag_kind='kde')
        head_tail = os.path.split(csv)
        print(f'Saving seaborn picture to {os.path.join(head_tail[0], "..", "Pictures", "sns_plotting.png")}')
        # Check for valid dir
        if not os.path.exists(os.path.join(head_tail[0], "..", "Pictures")):
            os.mkdir(os.path.join(head_tail[0], "..", "Pictures"))
        plt.savefig(os.path.join(head_tail[0], "..", "Pictures", "sns_plotting.png"), format='png')
        plt.close()
    except:
        raise Exception("Seaborn Plotting did not work (not essential for training)")

    feature_columns = []
    dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    print(dataset.shape)
    for feature_name in features:
        if dataset.loc[:, feature_name].dtype == "object":
            print(f"classifying {feature_name}:")
            vocabulary = dataset.loc[:, feature_name].unique()
            print(f"unique values are: {vocabulary}")


    # Encode non-numeric data with one-hot encoding
    dataset = pd.get_dummies(dataset, columns=categorical_features, prefix='', prefix_sep='')
    dataset.tail()
    print(dataset.shape)
    print(dataset.columns)
    print(dataset.dtypes)

    train_data, test_data = train_test_split(dataset, train_size=0.8, random_state=len(dataset))
    print(f"train data shape is {train_data.shape}")
    print(f"test data shape is {test_data.shape}")

    train_labels = train_data.pop("Total_Stiff")
    test_labels = test_data.pop("Total_Stiff")
    print(train_labels.shape)
    print(train_data.shape)
    print(train_data.describe().transpose()[['mean', 'std']])

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_data))
    print(normalizer.mean.numpy())
    # first = np.array(train_data[:1])
    #
    # with np.printoptions(precision=2, suppress=True):
    #   print('First example:', first)
    #   print()
    #   print('Normalized:', normalizer(first).numpy())

    model = build_and_compile_model(normalizer)
    model.summary()
    print(train_data)
    print(train_labels)
    batch_size = 32
    print(len(train_data))
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.2,
        verbose=2, epochs=1000,
        shuffle=True, batch_size=batch_size,
    )

    plot_loss(head_tail, history)
    score = model.evaluate(test_data, test_labels, verbose=2)
    print(f"model loss is: {score[0]}")
    print(f"model mean absolute error is: {score[1]}")
    print(f"model mean squared error is: {score[2]}")
    print(f"model mean average percent error is {score[3]}")

    # Plot predictions
    test_predictions = model.predict(test_data).flatten()
    plt.figure(3)
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True values (Stiffness)')
    plt.ylabel('Predictions (Stiffness)')
    lims = [0, 0.02]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", "stiffness_DNN_image_based.png"), format='png')

    plt.figure(4)
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Stiffness]')
    plt.ylabel('Count')
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", "image_based_error.png"), format='png')

if __name__ == "__main__":
    csv_path = r"F:\WorkData\MULTIS\master_csv\001_MasterList_indentation.csv"
    numeric_features = ["Total_Stiff", "Age", "BMI", "Thickness", "Skin", "Fat", "Muscle"]
    categorical_features = ["SubID", "Location", "Gender", "ActivityLevel", "Race", "Ethnicity"]
    tf_demographics(csv_path, categorical_features=categorical_features, numerical_features=numeric_features)