"""
Technique other than rejection sampling
everything is based on tensorflow 2.0
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from time import *
from termcolor import colored

tflogger = tf.get_logger()
tflogger.setLevel('ERROR')

# this is important or rejection sample won't work
# I set this here so that all random process keeps the same order but you can
# also set it at rejection_resample function
tf.random.set_seed(1341)

def compute_distribution(dataset, epochs, batch_size):
    data_receives = []
    # count stats to verify both batch distribution and total distribution
    bucket_counts = [0, 0]
    batch_counts = []
    for epoch in range(epochs):
        data_receives.append([])
        for data_contents in dataset:
            batch_count = [0, 0]
            for j in range(batch_size):
                try:
                    data_content = data_contents.numpy()[j]
                    data_receives[epoch].append(data_content)
                    bucket_counts[data_content] += 1
                    batch_count[data_content] += 1
                except IndexError:
                    break
            batch_counts.append(batch_count)

    # Batch distribution
    for bid, batch in enumerate(batch_counts):
        print("batch %03d, batch count [%02d, %02d], batch distribution [%02.2f, %02.2f]"
        % (bid, batch[0], batch[1], batch[0] / np.sum(batch), batch[1] / np.sum(batch)))

    # Total distribution
    print(colored("result counts [%d %d] final distribution [%02.2f, %02.2f]"
          % (bucket_counts[0], bucket_counts[1],
          bucket_counts[0] / np.sum(bucket_counts), bucket_counts[1] / np.sum(bucket_counts)),
          "green"))

    # check if each epoch produces the same order
    for i in range(len(data_receives)-1):
        if not np.array_equal(np.array(data_receives[i]), np.array(data_receives[i+1])):
            raise ValueError("Order not preserved.")
    if epochs > 1:
        print(colored("Yes, order is preserved.", "green"))

# generate data and distribution
np_data = np.array([
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

batch_size = 40
epochs = 1
target_dist = np.array([0.5, 0.5], dtype=np.float32)
number_of_positives = np.sum(np_data)
number_of_negatives = np_data.shape[0] - number_of_positives
number_of_examples = np_data.shape[0]
init_dist = np.array([number_of_negatives / number_of_examples, number_of_positives / number_of_examples], dtype=np.float32)
all_dataset = tf.data.Dataset.from_tensor_slices(np_data)


if __name__ == "__main__":
    # undersampling methods: my data is temporal, I'd prefer to read
    # and skip to undersample to get variety of different samples. The following two
    # are not tested and added, because of this reason.
    # TODO: Add tf.random.fixed_unigram_candidate_sampler & choose_from_dataset

    import sys
    if sys.argv[1] == "1":
        # Method 1: sample_from_dataset using weights !!Doesn't produce the right distribution
        # weights = fraction of samples you want to draw from the total number of samples of that class
        # usually for undersampled example, you should put 1., then the other should get
        # number of undersampled / total number in class to be drawn
        print(colored("Method 1: Use sample_from_dataset", "green"))
        dataset0 = all_dataset.filter(lambda x: tf.equal(x, 0))
        dataset1 = all_dataset.filter(lambda x: tf.equal(x, 1))
        list_of_datasets = [dataset0, dataset1]     # [class0, class1]
        method1_dataset = tf.data.experimental.sample_from_datasets(list_of_datasets, weights=[number_of_positives / number_of_negatives, 1], seed=2322)
        method1_dataset = method1_dataset.batch(batch_size)
        compute_distribution(method1_dataset, epochs, batch_size)

    elif sys.argv[1] == "2":

        # Method 2: dataset.filter based on distribution
        print(colored("Method 2: Filter based on uniform distribution", "green"))
        def filter(dist):
            # create a uniform distribution
            normal = tfp.distributions.Uniform(low=0, high=1.0)
            def _filter(sample):
                # explanation: uniform distribution has 60% chance larger than 0.4
                # so if you want to draw a sample at 60% chance, your tocken
                # needs to be larger than (1 - 60%)
                tocken = normal.sample()
                return tf.cast(tocken > (1 - dist[sample]), tf.bool)
            return _filter

        # Undersampling the first class, so directly take exhausted class 1
        # and have d1/d0 chance to take class 0
        dist = tf.constant([init_dist[1] / init_dist[0], 1], tf.float32)
        method2_dataset = all_dataset.filter(filter(dist))
        method2_dataset = method2_dataset.batch(batch_size)
        compute_distribution(method2_dataset, epochs, batch_size)

    # original counts and distribution
    print(colored("initial counts [%d %d] initial distribution [%2.2f, %2.2f]"
    % (number_of_negatives, number_of_positives, init_dist[0], init_dist[1]), "green"))
