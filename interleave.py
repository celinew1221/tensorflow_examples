import tensorflow as tf
from tensorflow.data import Dataset

a = Dataset.from_tensors((10, 10)).repeat(10)  # ==> [ 1, 2, 3, 4, 5 ]
b = Dataset.from_tensors([20, 20]).repeat(10)    # ==> [7, 8, 9, 10, 11]
dataset = Dataset.zip((a, b))

# NOTE: New lines indicate "block" boundaries.
dataset = dataset.interleave(lambda x, y: Dataset.from_tensors([x, y]),
                             cycle_length=12, block_length=2)
for data in dataset:
    print(data)
