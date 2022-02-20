# coding=utf-8
# This is a sample Python script.
import tensorflow as tf
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def basic_tf_variables():
    # Each tensor has a data type and shape (dimension)
    tf_string = tf.Variable("Seans string", tf.string)
    tf_number = tf.Variable(1234, tf.int16)
    tf_float = tf.Variable(3.567, tf.float64)

    rank1_tensor = tf.Variable(["one", "array", "one", "list"], tf.string)
    rank2_tensor = tf.Variable(["list in", "list"], ["rank", "two"], tf.string)

    tf.rank(rank1_tensor) #np1
    tf.rank(rank2_tensor) #np2

    rank1_tensor.shape # 4
    rank2_tensor.shape #[2,2]

    tensorone = tf.ones([1,2,3]) #1 list, 2 lists inside, 3 elements
    print(tensorone)
    tensortwo = tf.reshape(tensorone, [2,3,1]) #2 lists, 3 lists inside each list, 1 element each interior list
    tensorthree = tf.reshape(tensortwo,[3, -1]) #-1 calculate size of dimension for us [3,2]]! useful for us!
# Press the green button in the gutter to run the script.

    print(tensortwo)
    print(tensorthree)


    with tf.Session() as sess:
        tensorone.eval()


if __name__ == '__main__':
    basic_tf_variables()

