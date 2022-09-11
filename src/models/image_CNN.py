import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras_tuner
from keras import layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
import shap
import warnings
import random
import pydicom
from pydicom.pixel_data_handlers import convert_color_space
import cv2
from natsort import natsorted
import functools

# constant variables
IMAGE_SIZE = [676, 676]
COLOR_CHANNELS = 3


def ima_to_png(ima_dir: str, png_dir: str, df: pd.DataFrame(), plt_bool=False, matching_strs=[]) -> list:
    """function to convert ima files to png for compatibility with tf. tf only supports 4 image types officially
    @param ima_dir: where the raw ima files are located. Pull from multis gamma
    @param png_dir: where to save the converted files
    @param df: a pandas dataframe to be used to only pull specific subjects and regions that have valid data
    @param plt_bool: if true, will show a plot of how cropping is working
    @param matching_strs: which files are of interest? I was only focusing on anterior central and posterior central
                         indentation trials

    @return output: a list of files to be analyzed
    """
    output = []
    for root, _, files in os.walk(ima_dir):
        if files:
            wanted_regions = [x for x in files if any(val in x for val in matching_strs)]
            # filter out any non image files that may have been downloaded (or accidentally created)
            wanted_regions = [x for x in wanted_regions if ".IMA" in x]
            if wanted_regions:
                for file in wanted_regions:
                    # sample naming convention is: 023_MULTIS001-1_LA_AP_A-2_frm247.IMA
                    sub_id = int(file.split("MULTIS")[1][0:3])
                    location = file.split("-1_")[1][0:4]

                    if not df[(df['SubID'] == sub_id) & (df['Location'] == location)].empty:
                        # save as png for easier image augmentation
                        all_dcm = pydicom.read_file(os.path.join(root, file))
                        orig_img = convert_color_space(all_dcm.pixel_array, "YBR_FULL_422", "RGB")  # convert to RGB
                        height, width, _ = np.shape(orig_img)
                        # crop out text
                        img = orig_img[round(height * .07):round(height * .95), round(width * .08):round(width * .89)]
                        height, width, _ = np.shape(img)  # cropped dims

                        ecg_mask = cv2.inRange(img, (70, 200, 200), (255, 255, 255)) # finds most of the bottom line
                        ecg_mask[0:round(height * .9), :] = 0  # include bright data in the top 90% of the image
                        diff_img = np.ones_like(img)
                        diff_img = diff_img * np.reshape(ecg_mask, (height, width, 1))
                        final_img = cv2.subtract(img, diff_img)
                        final_img = cv2.resize(final_img, (height, height))
                        cv2.imwrite(os.path.join(png_dir, str(sub_id) + location + ".png"), final_img)
                        output.append(os.path.join(png_dir, str(sub_id) + location + ".png"))

                        # generates a lot of images, mainly for sanity check of data filtering or for saving one picture
                        if plt_bool:
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(orig_img)
                            ax[0].set_xlabel("original image")
                            ax[1].imshow(img)
                            ax[1].set_xlabel("cropped image")
                            ax[2].imshow(final_img)
                            ax[2].set_xlabel("final image (mask used)")
                            plt.show()
                    else:
                        # the first 5 subjects and any row with null are excluded to keep df consistent between models
                        pass
    return list(set(output))  # duplicates in my list?


def plot_loss(head_tail, history):
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
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"image_loss_history_{date}.png"))


def plot_test_predictions(head_tail, model, test_df, test_labels):
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

    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"compliance_CNN_image_{date}.png"), format='png')

    # Second plot of histogram mape
    plt.figure()
    error = abs(test_predictions - test_labels)/test_labels * 100
    plt.hist(error, bins=25, edgecolor='black')
    plt.xlabel('Absolute Percent Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"image_error_{date}.png"), format='png')


def build_and_compile_model(hp, lr):
    """Defines the input function to build a deep neural network for tensorflow"""

    # Define hyperparameters
    # units = hp.Int("units", min_value=32, max_value=512, step=64)
    # dropout = hp.Float("dropout", min_value=0.1, max_value=0.6, step=0.1)
    # lr = 1e-4  # hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])

    # Define model architecture
    model = tf.keras.Sequential()
    pretrained_model = tf.keras.applications.Xception(input_shape=[*IMAGE_SIZE, COLOR_CHANNELS], include_top=False)
    pretrained_model.trainable = False
    model.add(pretrained_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(64, activation='relu'))  # kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.Dense(1))
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(lr),
                  metrics=["mae", "mse", "mape"])
    model.summary()
    return model


def tf_image(csv, raw_image_dir, png_data_dir, categorical_features, numerical_features):#, interface=[False, False], shapley=False):
    """Build deep neural network from a csv file, and perform plotting functions and shapley/gradio outputs"""

    # Read in the master_csv file
    dataset = pd.read_csv(csv)
    head_tail = os.path.split(csv)
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
    # Order data how the files will be saved
    dataset["Name"] = dataset["SubID"].astype(str) + dataset["Location"]
    dataset = dataset.sort_values(["SubID", "Location"])
    dataset = dataset.set_index("Name")

    # Remove null values from the dataset
    dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all())
    dataset = dataset.reset_index(drop=True)

    print(dataset.head)  # Show first five entries and some columns
    print(dataset.shape)
    print(dataset.dtypes)

    # train_data, test_data = train_test_split(dataset, train_size=0.8, random_state=752)
    # print(f"train data shape is {train_data.shape}")
    # print(f"test data shape is {test_data.shape}")
    print(dataset.describe().transpose()[['mean', 'std']])

    # Pop off the compliance metric, which is the inverse our output

    # Pull and pre process .IMA files to .png for easier usage
    matching_regions = ["AC", "PC"]  # only interested in anterior central and posterior central trials
    output = ima_to_png(raw_image_dir, png_data_dir, dataset, plt_bool=False, matching_strs=matching_regions)
    output = natsorted(output)  # properly sorts numbers as 1,2,3 instead of 1, 10, 100, 2

    dataset["filepath"] = output

    # to visually inspect if filepath matches sub_id and region
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     pd.set_option('display.max_colwidth', None)
    #     print(dataset)

    batch_size = 4
    num_epochs = 100
    train, test = train_test_split(dataset, test_size=0.2, random_state=752)

    train, val = train_test_split(train, test_size=0.2, random_state=752)

    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,
                                       fill_mode="nearest", zoom_range=0.1,
                                       width_shift_range=0.1, height_shift_range=0.1,
                                       rotation_range=10, shear_range=10)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=train, directory=None, has_ext=True,
                                                        x_col="filepath", y_col="Compliance", class_mode="other",
                                                        target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                                        batch_size=batch_size, seed=752)

    val_generator = test_datagen.flow_from_dataframe(dataframe=val, directory=None, has_ext=True,
                                                     x_col="filepath", y_col="Compliance", class_mode="other",
                                                     target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                                     batch_size=batch_size, shuffle=False, seed=752)

    test_generator = test_datagen.flow_from_dataframe(dataframe=test, directory=None, has_ext=True,
                                                      x_col="filepath", y_col="Compliance", class_mode="other",
                                                      target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
                                                      batch_size=batch_size, shuffle=False, seed=752)
    # for _ in range(1):
    #     fig, axs = plt.subplots(1, 2)
    #     img, label = train_generator.next()
    #     print(img.shape)  # (batch size, IMAGESIZE, IMAGESIZE, COLORCHANNELS)
    #     axs[0].imshow(img[0])
    #     axs[1].imshow(img[1])
    #     axs[0].set_xlabel(f"compliance = {label[0]}")
    #     axs[1].set_xlabel(f"compliance = {label[1]}")
    #     plt.show()
    #
    # for _ in range(1):
    #     fig, axs = plt.subplots(1, 2)
    #     img, label = test_generator.next()
    #     print(img.shape)  # (batch size, IMAGESIZE, IMAGESIZE, COLORCHANNELS)
    #     axs[0].imshow(img[0])
    #     axs[1].imshow(img[1])
    #     axs[0].set_xlabel(f"compliance = {label[0]}")
    #     axs[1].set_xlabel(f"compliance = {label[1]}")
    #     plt.show()


    # create a log directory for a tensorboard visualization
    os.makedirs(os.path.join(head_tail[0], "..", "logs", "image_fit"), exist_ok=True)
    log_path = os.path.join(head_tail[0], "..", "logs", "image_fit")
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=(train_generator.samples/batch_size)*5,
        decay_rate=0.9,
        staircase=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(log_path, monitor='val_loss', verbose=1, save_freq='epoch',
                                                          save_weights_only=True, save_best_only=True, mode='min',
                                                          restore_best_weights=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    model = build_and_compile_model(hp=None, lr=1e-3)

    # define hyperparameters through keras tuner
    #hp = keras_tuner.HyperParameters()

    # define tuning model for hyperparameter search
    # tuner = keras_tuner.tuners.RandomSearch(
    #     hypermodel=build_and_compile_model(hp),
    #     objective='val_loss',
    #     max_trials=10,
    #     executions_per_trial=2,
    #     overwrite=True,
    #     directory=log_path,
    #     project_name="TestingKerasTuner")
    # tuner.search_space_summary()

    # The call to search has the same signature as model.fit()
    # tuner.search(train_data,
    #              train_labels,
    #              validation_split=0.2,
    #              verbose=2, epochs=1000,
    #              shuffle=True, batch_size=batch_size,
    #              callbacks=[early_stop, model_checkpoint, tensorboard_callback, reduce_lr]
    # )
    # # retrieve best result and visualize top values
    # tuner.results_summary()
    # best_hp = tuner.get_best_hyperparameters()[0]  # get the best hp and refit model for plotting
    # model = tuner.hypermodel.build(best_hp)

    history = model.fit(
        train_generator, verbose=2, steps_per_epoch=train_generator.samples / batch_size,
        validation_data=val_generator, validation_steps=val_generator.samples / batch_size,
        epochs=num_epochs, shuffle=True, callbacks=[model_checkpoint, reduce_lr, early_stop, tensorboard_callback]
    )
    # visualize model loss
    plot_loss(head_tail, history)

    # print out metrics on how the model performed on the test data
    score = model.evaluate(test_generator, verbose=2)
    print(f"model loss is: {score[0]}")
    print(f"model mean absolute error is: {score[1]}")
    print(f"model mean squared error is: {score[2]}")
    print(f"model mean average percent error is {score[3]}")

    # Plot predictions vs true values to visually assess accuracy and histogram of APE distribution
    plot_test_predictions(head_tail, model, test_generator, test["Compliance"])

    # # Plot different shapley figures for feature importance
    # if shapley:
    #     print(test_data.iloc[0])
    #     background = train_data.head(100)  # data is already shuffled, no need to randomly choose?
    #
    #     # hide the sklearn future deprecation warning as it clogs the terminal
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings("ignore")
    #         explainer = shap.KernelExplainer(model, background)
    #         shap_values = explainer.shap_values(test_data)
    #
    #     # visualize first test data point: https://github.com/slundberg/shap/issues/1420. waterfall legacy seems to plot
    #     fig = plt.figure()
    #     fig = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0][0], test_data.iloc[0],
    #                                            max_display=20, show=False)
    #     date = datetime.now().strftime("%Y_%m_%d-%I%p")
    #     print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_waterfall_{date}.png")}')
    #     plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_waterfall_{date}.png"), format='png')
    #     plt.close()
    #
    #
    #     # Generate a force plot for the same entry
    #     fig2 = plt.figure()
    #     fig2 = shap.force_plot(explainer.expected_value[0], shap_values[0][0], test_data.iloc[0],
    #                            matplotlib=True, show=False)
    #     print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_force_{date}.png")}')
    #     plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_force_{date}.png"), format='png')
    #     plt.close()
    #
    #     # Generate a summary plot for all entries
    #     fig3 = plt.figure()
    #     fig3 = shap.summary_plot(shap_values, train_data, show=False)
    #     print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_summary_{date}.png")}')
    #     plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_summary_{date}.png"), format='png')
    #     plt.close()
    #
    #     # Generate a force plot for the same entry
    #     fig4 = plt.figure()
    #     fig4 = shap.summary_plot(shap_values[0], features=test_data.columns, show=False)
    #     print(f'Saving shap picture to {os.path.join(head_tail[0], "..", "Pictures", f"shapley_beeswarm_{date}.png")}')
    #     plt.savefig(os.path.join(head_tail[0], "..", "Pictures", f"shapley_beeswarm_{date}.png"), format='png')
    #     plt.close()
    #
    # # Generate a gradio web interface if requested
    # if interface[0]:
    #     import gradio as gr
    #
    #     def make_prediction(*features):
    #         """Function to make a prediction on a new user value from a gradio user interface"""
    #         Compliance = 1000  # Dummy val just for the one hot encoder. Probably a more graceful solution available?
    #         features = list(features)
    #         features.append(Compliance)
    #         cols = categorical_features + numerical_features
    #         pred_df = pd.DataFrame(data=[features], columns=cols)
    #
    #         # Convert user input to one hot encoded values so predictions can be made
    #         ohe_pred = ohe_df(enc, pred_df, categorical_features)
    #         ohe_pred.drop(["Compliance"], axis=1, inplace=True)
    #
    #         # Sanity check that variables look correct
    #         with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #             print(ohe_pred)
    #         pred = model.predict([ohe_pred])
    #
    #         print(f"Predicted tissue compliance of {pred[0][0]} mm/MPa")
    #         return f"Predicted tissue compliance of {pred[0][0]} mm/MPa"
    #
    #     input_list = []
    #     #  Create dropdown menus for categorical inputs
    #     for variable in categorical_features:
    #         if variable != "SubID":
    #             if dataset.loc[:, variable].dtype == "object":
    #                 vocabulary = list(dataset.loc[:, variable].unique())
    #                 input_list.append(gr.inputs.Dropdown(label=f"Choose a value for {variable}", choices=vocabulary))
    #         else:
    #             input_list.append(gr.inputs.Textbox(label=f"Choose a value for {variable} (Enter 999 or higher for new subjects)"))
    #
    #     #  Create number menus for numerical inputs
    #     for variable in numerical_features:
    #         if variable != "Compliance":
    #             min_val = min(train_data[variable])
    #             max_val = max(train_data[variable])
    #             input_list.append(gr.inputs.Number(label=f"Choose a value for {variable}, with acceptable range from {min_val} to {max_val}"))
    #         else:
    #             pass  # dummy value in make_prediction just to satisfy one hot encoder. Better solution possible?
    #
    #     #  Launch gradio interface on the web
    #     app = gr.Interface(fn=make_prediction, inputs=input_list, outputs="text")
    #     app.launch(share=interface[1]) # share=True to display on the gradio webpage. Can share on huggingfaces


if __name__ == "__main__":
    csv_path = r"F:\WorkData\MULTIS\master_csv\001_MasterList_indentation_orig.csv"
    raw_image_dir = r"F:\WorkData\MULTIS\Ultrasound_minImages"
    png_dir = r"F:\WorkData\MULTIS\master_csv\ultrasound_tiff" # as defined by keras image_flow from dir
    categorical_features = ["SubID", "Location"]
    numeric_features = ["Total_Stiff"]
    #  lists are sorted alphabetically so order does not matter

    make_interface = [True, False]  # 1st bool - make gradio interface | second bool - share to the web
    tf_image(csv_path, raw_image_dir, png_dir, categorical_features, numeric_features) #, interface=make_interface), shapley=plot_shapley)
