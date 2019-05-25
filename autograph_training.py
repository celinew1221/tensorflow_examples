import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Progbar
from tqdm import tqdm
import numpy as np
from time import perf_counter
import sys
import logging
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

# Load a toy dataset for the sake of this example
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


def get_uncompiled_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
  x = layers.Dense(64, activation='relu', name='dense_2')(x)
  outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


def get_compiled_model():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
  return model


@tf.function
def update_metrics_state(metric_list, y_batch, loss):
    for metric in metric_list:
        metric.update_state(y_batch, loss)


@tf.function
def reset_metrics(metric_list):
    for metric in metric_list:
        metric.reset_states()


@tf.function
def log_metrics(metric_list, step, prefix=""):
    for metric in metric_list:
        tf.summary.scalar("{}_{}".format(prefix, metric.name), metric.result(), step=step)


@tf.function
def train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
      logits = model(x_batch_train)  # Logits for this minibatch
      loss_value = loss_fn(y_batch_train, logits)
      loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value, logits


@tf.function
def train(train_dataset, model, optimizer, loss_fn, metrics):
    step = 0
    for e in range(1):
        for data in (train_dataset):
            step += 1
            x_batch_train, y_batch_train = data["x"], data["y"]
            loss_value, logits = train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train)
            update_metrics_state(metrics, y_batch_train, logits)
            if tf.equal(step % 200, 0):
                tf.print('Epoch {} Training loss at step'.format(e), step, float(loss_value))
            log_metrics(metrics, step=optimizer.iterations, prefix="batch")
        log_metrics(metrics, step=e, prefix="epoch")
        reset_metrics(metrics)
        tf.summary.text("Time Elapsed", "Last {}".format(perf_counter()-s), step=e, description="time")

s = perf_counter()
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
with strategy.scope():
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    train_dataset = train_dataset.map(lambda x,y: {"x": x, "y":y})
    model = get_uncompiled_model()
    optimizer = keras.optimizers.RMSprop(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)
    metrics = [tf.metrics.SparseCategoricalCrossentropy(name="loss")]

    summary = tf.summary.create_file_writer("./log")
    with summary.as_default():
        train(train_dataset, model, optimizer, loss_fn, metrics)
print(perf_counter() - s)
