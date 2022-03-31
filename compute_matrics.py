import tensorflow as tf
import pandas as pd
import numpy as np


def mean_absolute_scaled_error(real, pred):

  mae = tf.reduce_mean(tf.abs(real - pred))

  mae_naive_no_season = tf.reduce_mean(tf.abs(real[1:] - real[:-1]))

  return mae / mae_naive_no_season


def compute_metrics(actual, predictions):
    predictions = tf.reshape(predictions,shape=(len(predictions),))
    actual = tf.reshape(actual, shape=(len(actual),))

    mae = tf.keras.metrics.mean_absolute_error(actual, predictions)
    mse = tf.keras.metrics.mean_squared_error(actual, predictions)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(actual, predictions)

    return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy()}
