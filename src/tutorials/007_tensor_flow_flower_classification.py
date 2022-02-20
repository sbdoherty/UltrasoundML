import tensorflow as tf
import pandas as pd

def classification_example():
    """Simple classification ML example in TF. Starts with loading data set, moves to visualizing data set and
       ends with using ML model to classify flowers by species.
    """
    #describe data
    csv_column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    species = ['Setosa', 'Versicolor', 'Virginica']

    #What does keras do? -> deep learning api
    train_path = tf.keras.utils.get_file(fname=None,
                                         origin='https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv') #training data
    test_path = tf.keras.utils.get_file(fname=None,
                                        origin='https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv') #training data

    train = pd.read_csv(train_path, names=csv_column_names, header=0)
    test = pd.read_csv(test_path, names=csv_column_names, header=0)
    print(train.head()) #show first five entries and some columns

    y_train = train.pop('Species') #species vector as y_train
    y_test = test.pop('Species')
    print(train.shape) #120 rows, 4 columns
    train.describe() #gives some statistics and entry count on data

    #Features are needed to describe input
    feature_columns = []
    for key in train.keys(): #train.keys() returns column names for pd
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    print(feature_columns)

    #Let's build a classifier model!
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        #two hidden layers of 30 and 10 nodes (?)
        hidden_units=[30, 10],
        #model must choose between 3 classes
        n_classes=3
    )

    #x = lambada: print('hi')
    #x() -> hi
    classifier.train(
        # steps is similar to epochs, continue until we have looked at 5000 items
        input_fn=lambda: make_input_fxn(train, y_train, training=True), steps=5000
    )

    eval_result = classifier.evaluate(
        input_fn=lambda: make_input_fxn(test, y_test, training=False))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result)
    )


    def input_fn(features, batch_size=256):
        # convert inputs to dataset without labels
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


    features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    predict = {}

    for feature in features:
        valid = True
        while valid:
            val = input(feature + ": ")
            if not val.isdigit(): valid = False
        predict[feature] = [float(val)]
    predictions = classifier.predict(input_fn=lambda: input_fn(predict))
    for pred_dict in predictions:
        print(pred_dict)
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(species[class_id], 100 * probability))



def make_input_fxn(features, labels, training=True, batch_size=256):
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels)) #create tf.data.Dataset object with data and its label
    if training: #when in training mode, shuffle and repeat data
        ds = ds.shuffle(1000).repeat() #randomize
    ds = ds.batch(batch_size) #split dataset into batches of 32, no epochs for this problem (?)
    return ds #batch of the dataset




if __name__ == '__main__':
    classification_example()


