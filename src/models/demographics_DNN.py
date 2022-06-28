import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime


def ohe_df(enc, df, columns):
    """function to convert a df to a one hot encoded df with appropriate column labels"""
    transformed_array = enc.transform(df)
    initial_colnames_keep = list(set(df.columns.tolist()) - set(columns))
    initial_colnames_keep.sort()
    new_colnames = []
    new_colnames = np.concatenate(enc.named_transformers_['OHE'].categories_).tolist()
    all_colnames = new_colnames + initial_colnames_keep
    df = pd.DataFrame(transformed_array, index=df.index, columns=all_colnames)
    return df


def plot_loss(head_tail, history):
    """Visualize error decrease over training process """
    plt.figure(2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    date = datetime.now().strftime("%Y_%m_%d-%I%p")

    if not os.path.exists(os.path.join(head_tail[0], "..", "Pictures")):
        os.mkdir(os.path.join(head_tail[0], "..", "Pictures"))
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"demographics_loss_history_{date}.png"))


def plot_test_predictions(head_tail, model, test_df, test_labels):
    """Function to visualize the quality of predictions on the test data"""
    test_predictions = model.predict(test_df).flatten()
    plt.figure()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True values (Stiffness)')
    plt.ylabel('Predictions (Stiffness)')
    lims = [0, 0.02]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    date = datetime.now().strftime("%Y_%m_%d-%I%p")

    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"stiffness_DNN_demographics_{date}.png"), format='png')

    # Second plot of histogram mape
    plt.figure()
    error = abs(test_predictions - test_labels)/test_labels * 100
    plt.hist(error, bins=25)
    plt.xlabel('Absolute Percent Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"demographics_error_{date}.png"), format='png')


def build_and_compile_model(norm):
    """Defines the input function to build a deep neural network for tensor flow"""
    model = keras.Sequential([
      norm,
      layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
      layers.Dropout(0.5),
      layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
      layers.Dropout(0.5),
      layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.0001),
                  metrics=["mae", "mse", "mape"])
    return model


def tf_demographics(csv, categorical_features, numerical_features="", interface=False):
    # Read in the master_csv file
    dataset = pd.read_csv(csv)
    categorical_features.sort()
    numerical_features.sort()  # need sorted numerical features to reconstruct the df
    features = [*categorical_features, *numerical_features]

    # Filter the csv by the desired features
    dataset = dataset[features]
    dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    dataset = dataset.dropna()
    print(dataset.head())  # Show first five entries and some columns
    print(dataset.shape)
    print(dataset.dtypes)

    # Plotting of numeric variables, not necessary for DNN
    if numerical_features:
        plt.figure()
        sns.pairplot(dataset[numerical_features], diag_kind='kde')
        head_tail = os.path.split(csv)
        date = datetime.now().strftime("%Y_%m_%d-%I%p")
        print(f'Saving seaborn picture to {os.path.join(head_tail[0], "..", "Pictures", f"sns_plotting_{date}.png")}')

        # Check for valid dir
        if not os.path.exists(os.path.join(head_tail[0], "..", "Pictures")):
            os.mkdir(os.path.join(head_tail[0], "..", "Pictures"))
        plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"sns_plotting_{date}.png"), format='png')
        plt.close()


    # Remove null values from the dataset
    dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    dataset = dataset.reset_index(drop=True)

    #  Export to a clean csv if you want to do further analysis outside of python
    # dataset.to_csv(r"F:\WorkData\MULTIS\master_csv\001_MasterList_weka.csv")

    #  List out unique values in each variable passed into the model
    for feature_name in features:
        if dataset.loc[:, feature_name].dtype == "object":
            print(f"classifying {feature_name}:")
            vocabulary = dataset.loc[:, feature_name].unique()
            print(f"unique values are: {vocabulary}")

    # Encode non-numeric data with one-hot encoding
    enc = ColumnTransformer([("OHE", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features)],
                            remainder="passthrough")
    enc.fit(dataset)  # fit the encoder to the datasets variables

    # call function to onehot encode the data
    ohe_dataset = ohe_df(enc, dataset, categorical_features)

    # Split training and testing data
    train_data, test_data = train_test_split(ohe_dataset, train_size=0.8, random_state=len(dataset))
    print(f"train data shape is {train_data.shape}")
    print(f"test data shape is {test_data.shape}")
    print(train_data.describe().transpose()[['mean', 'std']])

    # Pop off the Total stiffness metric, which is our label
    train_labels = train_data.pop("Total_Stiff")
    test_labels = test_data.pop("Total_Stiff")

    #  Normalize the inputs
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_data))
    # print(normalizer.mean.numpy()) # if nan the model wont train

    # # Visualization of normalized variables, mostly unneccesary
    # first = np.array(train_data[:1])
    #
    # with np.printoptions(precision=2, suppress=True):
    #   print('First example:', first)
    #   print()
    #   print('Normalized:', normalizer(first).numpy())

    # Build the model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    model = build_and_compile_model(normalizer)
    model.summary()
    batch_size = 32

    # create a log directory for a tensorboard visualization
    if not os.path.exists(os.path.join(head_tail[0], "..", "logs", "fit")):
        os.mkdir(os.path.join(head_tail[0], "..", "logs", "fit"))


    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.2,
        verbose=2, epochs=1000,
        shuffle=True, batch_size=batch_size,
    )
    # visualize model loss
    plot_loss(head_tail, history)

    # print out metrics on how the model performed on the test data
    score = model.evaluate(test_data, test_labels, verbose=2)
    print(f"model loss is: {score[0]}")
    print(f"model mean absolute error is: {score[1]}")
    print(f"model mean squared error is: {score[2]}")
    print(f"model mean average percent error is {score[3]}")

    # Plot predictions vs true values to visually assess accuracy and histogram of APE distribution
    plot_test_predictions(head_tail, model, test_data, test_labels)

    # Generate a gradio web interface if the user requested it
    if interface:
        import gradio as gr

        def make_prediction(*features):
            """Function to make a prediction on a new user value from a gradio user interface"""
            Total_Stiff = 999  # Dummy val just for the one hot encoder. Probably a more graceful solution available?
            features = list(features)
            features.append(Total_Stiff)
            cols = categorical_features + numerical_features
            pred_df = pd.DataFrame(data=[features], columns=cols)

            # Convert user input to one hot encoded values so predictions can be made
            ohe_pred = ohe_df(enc, pred_df, categorical_features)
            ohe_pred.drop(["Total_Stiff"], axis=1, inplace=True)

            # Sanity check that variables look correct
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(ohe_pred)
            pred = model.predict([ohe_pred])

            print(f"Predicted tissue stiffness of {pred[0][0]} MPa/mm")
            return f"Predicted tissue stiffness of {pred[0][0]} MPa/mm"

        input_list = []
        #  Create dropdown menus for categorical inputs
        for variable in categorical_features:
            if variable != "SubID":
                if dataset.loc[:, variable].dtype == "object":
                    vocabulary = list(dataset.loc[:, variable].unique())
                    input_list.append(gr.inputs.Dropdown(label=f"Choose a value for {variable}", choices=vocabulary))
            else:
                input_list.append(gr.inputs.Textbox(label=f"Choose a value for {variable} (Enter 999 or higher for new subjects)"))

        #  Create number menus for numerical inputs
        for variable in numerical_features:
            if variable != "Total_Stiff":
                min_val = min(train_data[variable])
                max_val = max(train_data[variable])
                print(f"Choose a value for {variable}, with acceptable range from {min_val} to {max_val}")
                input_list.append(gr.inputs.Number(label=f"Choose a value for {variable}, with acceptable range from {min_val} to {max_val}"))
            else:
                pass  # dummy value in make_prediction just to satisfy one hot encoder. Better solution possible?

        #  Launch gradio interface on the web
        app = gr.Interface(fn=make_prediction, inputs=input_list, outputs="text")
        app.launch(share=False) # share=True to display on the gradio webpage. Can share on huggingfaces


if __name__ == "__main__":
    csv_path = r"F:\WorkData\MULTIS\master_csv\001_MasterList_indentation.csv"
    #  lists are sorted alphabetically so order does not matter
    categorical_features = ["SubID", "Location", "Gender", "ActivityLevel", "Race", "Ethnicity"]  # Features that aren't numerical
    numeric_features = ["Total_Stiff", "Age", "BMI"]  # Features defined by a range of numbers
    make_interface = False
    tf_demographics(csv_path, categorical_features=categorical_features, numerical_features=numeric_features, interface=make_interface)