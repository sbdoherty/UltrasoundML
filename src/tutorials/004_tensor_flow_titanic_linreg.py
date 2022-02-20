import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

def linear_regression_example():
    """Simple linear regression ML example in TF. Starts with loading data set, moves to visualizing data set and
       ends with using ML model.
    """
    # load data with survival, sex, age, n_siblings, parch, fare, class, deck, embark_to, alone
    # what correlations might exist with this data
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training data
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #testing data
    print(dftrain.head()) #show first five entries and some columns

    y_train = dftrain.pop('survived') #survive vector as y_train
    y_eval = dfeval.pop('survived')
    print(dftrain.head()) #popped survivor column
    print(dftrain.loc[0], y_train.loc[0]) #locate a specific item
    print(dftrain["age"]) #access age column

    dftrain.describe() #gives some statistics and entry count on data
    dftrain.shape #627 rows, 9 features (columns)

    #generate visuals
    dftrain.age.hist(bins=20)
    dftrain.sex.value_counts().plot(kind='barh')
    dftrain['class'].value_counts().plot(kind='barh')
    pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

    dfeval.shape #only 264 rows, less than training set

    #Describe data types. Convert categorical to binary representation
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERICAL_COLUMNS = ['age', 'fare']

    #Features are needed to run a linear regression
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique() #list of all unique vals from feature column
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERICAL_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    train_input_fxn = make_input_fxn(dftrain, y_train)
    eval_input_fxn = make_input_fxn(dfeval, y_eval, num_epochs=1, shuffle=False)
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

    linear_est.train(train_input_fxn)
    result = linear_est.evaluate(eval_input_fxn) #dict of stats
    result = list(linear_est.predict(eval_input_fxn)) #result is a generator object, meant to be looped through

    print(dfeval.loc[0])
    print(y_eval.loc[0])
    print(result[0].get('probabilities')[1]) #what is probability first person in list survived

def make_input_fxn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000) #randomize
        ds = ds.batch(batch_size).repeat(num_epochs) #split dataset into batches ofo 32, repeat for num of epochs
        return ds #batch of the dataset
    return input_function #return a function object!


    print(feature_columns)
if __name__ == '__main__':
    linear_regression_example()


