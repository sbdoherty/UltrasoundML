import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

def hidden_markov_model():
    """Simple HMM about weather based on TF tutorial. Cold days = 0, hot day = 1.
    First day in sequence has 80% chance of being cold.
    Cold day has 30% chance of being followed by a hot day
    "hot day has  20% chance of being followed by cold day. """

    tfd = tfp.distributions #shortcut for making distributions
    initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
    transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                     [0.2, 0.8]])
    #loc provides mean, scale is SD
    observation_distribution = tfd.Normal(loc=[0.0, 15.0], scale=[5.0, 10.0])

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=7) # num_steps -> how many days we need to predict
    mean = model.mean()
    print(mean)

    with tf.compat.v1.Session() as sess:
        print(mean.numpy())


if __name__ == '__main__':
    hidden_markov_model()


