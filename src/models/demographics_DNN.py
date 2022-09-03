import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functools import partial
import tensorflow as tf
import keras_tuner
from keras import layers, regularizers
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
import shap
import warnings


def ohe_df(enc: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """function to convert a df to a one hot encoded df with appropriate column labels"""
    transformed_array = enc.transform(df)
    initial_colnames_keep = list(df.select_dtypes(include=np.number).columns)  # essentially the numeric labels
    new_colnames = np.concatenate(enc.named_transformers_['OHE'].categories_).tolist()   # unique category classes
    all_colnames = new_colnames + initial_colnames_keep
    df = pd.DataFrame(transformed_array, index=df.index, columns=all_colnames)

    return df


def plot_loss(head_tail: list, history: tf.keras.callbacks.History):
    """Visualize error decrease over training process """
    plt.figure(2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    date = datetime.now().strftime("%Y_%m_%d-%I%p")

    if not os.path.exists(os.path.join(head_tail[0], "..", "Pictures")):
        os.mkdir(os.path.join(head_tail[0], "..", "Pictures"))
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"demographics_loss_history_{date}.png"))


def plot_test_predictions(head_tail: list, model: tf.keras.Model, test_df: pd.DataFrame, test_labels: pd.DataFrame):
    """Function to visualize the quality of predictions on the test data"""
    test_predictions = model.predict(test_df).flatten()
    plt.figure()
    plt.scatter(test_predictions, test_labels)

    plt.xlabel('Predicted Compliance (mm/MPa)')
    plt.ylabel('Experimental Compliance (mm/MPa)')
    lims = [0, 3500]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims, linestyle='dashed', color='black', linewidth=2)
    date = datetime.now().strftime("%Y_%m_%d-%I%p")

    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"compliance_DNN_demographics_{date}.png"), format='png')

    # Second plot of histogram mape
    plt.figure()
    error = abs(test_predictions - test_labels)/test_labels * 100
    plt.hist(error, bins=25, edgecolor='black')
    plt.xlabel('Absolute Percent Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"demographics_error_{date}.png"), format='png')


def build_and_compile_model(hp: keras_tuner.HyperParameters, norm: tf.keras.layers.Normalization) -> tf.keras.Sequential:
    """Defines the input function to build a deep neural network for tensorflow"""

    # Define hyperparameters
    units = hp.Int("units", min_value=32, max_value=512, step=64)
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.6, step=0.1)
    lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])
    #l2 = hp.Float("l2", min_value=1e-5, max_value=1e-3, sampling="log")
    gaussian_noise = hp.Boolean("gaussian_noise")

    # Define model architecture
    model = tf.keras.Sequential()
    model.add(norm)
    if gaussian_noise:
        model.add(layers.GaussianNoise(0.1))
    model.add(layers.Dense(units=units))#, kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.PReLU(alpha_initializer=tf.initializers.constant(0.1)))
    model.add(layers.Dropout(rate=dropout))
    model.add(layers.Dense(units=units))#, kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.PReLU(alpha_initializer=tf.initializers.constant(0.1)))
    model.add(layers.Dropout(rate=dropout))
    model.add(layers.Dense(1))
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(lr),
                  metrics=["mae", "mse", "mape"])
    model.summary()
    return model


def tf_demographics(csv: str, categorical_features: list, numerical_features: list, interface=[False, False], shapley=False):
    """Build deep neural network from a csv file, and perform plotting functions and shapley/gradio outputs
       csv: path to the data file
       categorical features: non numeric feature names that will be pulled from the csv into the dataframe
       numerical features: numeric feature names that will be pulled from the csv into dataframe
       interface: first bool: do you want gradio at all? second bool: do you want to push gradio app to web?
       shapley: do you want to try out feature importance plots? """

    # Read in the master_csv file
    dataset = pd.read_csv(csv)
    categorical_features.sort()
    numerical_features.sort()  # need sorted numerical features to reconstruct the df
    features = [*categorical_features, *numerical_features]

    # Filter the csv by the desired features
    dataset = dataset[features]
    dataset = dataset.rename(columns={'Total_Stiff': 'Compliance'})
    dataset['Compliance'] = 1/dataset['Compliance']

    i = numerical_features.index("Total_Stiff")
    numerical_features[i] = "Compliance"
    i = features.index("Total_Stiff")
    features[i] = "Compliance"

    dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    dataset = dataset.dropna()
    print(dataset.head())  # Show first five entries and some columns
    print(dataset.shape)
    print(dataset.dtypes)

    # Plotting of numeric variables, not necessary for DNN
    head_tail = os.path.split(csv)
    if numerical_features:
        plt.figure()
        sns.pairplot(dataset[numerical_features], diag_kind='kde')
        date = datetime.now().strftime("%Y_%m_%d-%I%p")
        print(f'Saving seaborn picture to {os.path.join(head_tail[0], "..", "Pictures", f"sns_plotting_{date}.png")}')

        # Check for valid dir
        os.makedirs(os.path.join(head_tail[0], "..", "Pictures"), exist_ok=True)
        plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"sns_plotting_{date}.png"), format='png')
        plt.close()

    # Remove null values from the dataset
    dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    dataset = dataset.reset_index(drop=True)
    # dataset.to_csv(r"F:\WorkData\MULTIS\master_csv\001_MasterList_weka.csv") # Export to clean cs

    #  List out unique values in each variable passed into the model. Just a sanity check and inspection of data
    for feature_name in features:
        if dataset.loc[:, feature_name].dtype == "object":
            print(f"classifying {feature_name}:")
            vocabulary = dataset.loc[:, feature_name].unique()
            print(f"unique values are: {vocabulary}")

    # Encode non-numeric data with one-hot encoding
    enc = ColumnTransformer([("OHE", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features)],
                            remainder="passthrough")
    enc.fit(dataset)  # fit the encoder to the datasets variables
    print(type(enc))

    # call function to onehot encode the data
    ohe_dataset = ohe_df(enc, dataset) #, categorical_features)

    # Split training and testing data
    train_data, test_data = train_test_split(ohe_dataset, train_size=0.8, random_state=752)
    print(f"train data shape is {train_data.shape}")
    print(f"test data shape is {test_data.shape}")
    print(train_data.describe().transpose()[['mean', 'std']])

    # Pop off the compliance metric, which is the inverse our output
    train_labels = train_data.pop("Compliance")
    test_labels = test_data.pop("Compliance")

    #  Normalize the inputs
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_data))
    # print(normalizer.mean.numpy()) # if nan the model will not train

    # create a log directory for a tensorboard visualization
    os.makedirs(os.path.join(head_tail[0], "..", "logs", "fit"), exist_ok=True)
    log_path = os.path.join(head_tail[0], "..", "logs", "fit")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(log_path, monitor='val_loss', verbose=2,
                                                          save_weights_only=True, save_best_only=True, mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=50)

    # define hyperparameters through keras tuner
    hp = keras_tuner.HyperParameters()
    batch_size = 32

    # define tuning model for hyperparameter search
    build_model = partial(build_and_compile_model, norm=normalizer)
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory=log_path,
        project_name="TestingKerasTuner")
    tuner.search_space_summary()

    # The call to search has the same signature as model.fit()
    tuner.search(train_data,
                 train_labels,
                 validation_split=0.2,
                 verbose=2, epochs=1000,
                 shuffle=True, batch_size=batch_size,
                 callbacks=[early_stop, model_checkpoint, tensorboard_callback, reduce_lr]
    )
    # retrieve best result and visualize top values
    tuner.results_summary()
    best_hp = tuner.get_best_hyperparameters()[0]  # get the best hp and refit model for plotting
    model = tuner.hypermodel.build(best_hp)

    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=1000,
        shuffle=True, batch_size=32,
        callbacks=[early_stop, model_checkpoint, tensorboard_callback, reduce_lr]
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

    # Plot different shapley figures for feature importance
    if shapley:
        print(test_data.iloc[0])
        background = train_data.head(100)  # data is already shuffled, no need to randomly choose?

        # hide the sklearn future deprecation warning as it clogs the terminal
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            explainer = shap.KernelExplainer(model, background)
            shap_values = explainer.shap_values(test_data)

        # visualize first test data point: https://github.com/slundberg/shap/issues/1420. waterfall legacy seems to plot
        fig = plt.figure()
        fig = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][0], test_data.iloc[0],
                                               max_display=20, show=False)
        date = datetime.now().strftime("%Y_%m_%d-%I%p")
        print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_waterfall_{date}.png")}')
        plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_waterfall_{date}.png"), format='png')
        plt.close()


        # Generate a force plot for the same entry
        fig2 = plt.figure()
        fig2 = shap.force_plot(explainer.expected_value[0], shap_values[0][0], test_data.iloc[0],
                               matplotlib=True, show=False)
        print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_force_{date}.png")}')
        plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_force_{date}.png"), format='png')
        plt.close()

        # Generate a summary plot for all entries
        fig3 = plt.figure()
        fig3 = shap.summary_plot(shap_values, train_data, show=False)
        print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_summary_{date}.png")}')
        plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_summary_{date}.png"), format='png')
        plt.close()

        # Generate a force plot for the same entry
        fig4 = plt.figure()
        fig4 = shap.summary_plot(shap_values[0], features=test_data.columns, show=False)
        print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_beeswarm_{date}.png")}')
        plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_beeswarm_{date}.png"), format='png')
        plt.close()

    # Generate a gradio web interface if requested
    if interface[0]:
        import gradio as gr

        def make_prediction(*features):
            """Function to make a prediction on a new user value from a gradio user interface"""
            Compliance = 1000  # Dummy val just for the one hot encoder. Probably a more graceful solution available?
            features = list(features)
            features.append(Compliance)
            cols = categorical_features + numerical_features
            pred_df = pd.DataFrame(data=[features], columns=cols)

            # Convert user input to one hot encoded values so predictions can be made
            ohe_pred = ohe_df(enc, pred_df, categorical_features)
            ohe_pred.drop(["Compliance"], axis=1, inplace=True)

            # Sanity check that variables look correct
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(ohe_pred)
            pred = model.predict([ohe_pred])

            print(f"Predicted tissue compliance of {pred[0][0]} mm/MPa")
            return f"Predicted tissue compliance of {pred[0][0]} mm/MPa"

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
            if variable != "Compliance":
                min_val = min(train_data[variable])
                max_val = max(train_data[variable])
                input_list.append(gr.inputs.Number(label=f"Choose a value for {variable}, with acceptable range from {min_val} to {max_val}"))
            else:
                pass  # dummy value in make_prediction just to satisfy one hot encoder. Better solution possible?

        #  Launch gradio interface on the web
        app = gr.Interface(fn=make_prediction, inputs=input_list, outputs="text")
        app.launch(share=interface[1]) # share=True to display on the gradio webpage. Can share on huggingfaces


if __name__ == "__main__":
    csv_path = r"F:\WorkData\MULTIS\master_csv\001_MasterList_indentation_orig.csv"
    #  lists are sorted alphabetically so order does not matter
    categoric_features = ["Location", "Gender", "ActivityLevel", "Race", "Ethnicity"] # Features that aren't numerical
    numeric_features = ["Total_Stiff", "Age", "BMI"]  # Features defined by a range of numbers
    make_interface = [True, False]  # 1st bool - make a gradio second bool - share to the web
    plot_shapley = True  # takes a while to run, but generates feature importance plots
    tf_demographics(csv_path, categoric_features, numeric_features, interface=make_interface, shapley=plot_shapley)