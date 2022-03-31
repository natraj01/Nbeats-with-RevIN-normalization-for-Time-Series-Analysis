import tensorflow as tf
import pandas as pd
import numpy as np



def get_labelled_windows(x, horizon=1):
  return x[:, :-horizon], x[:, -horizon:]



def make_windows(x, window_size=10, horizon=1):
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T
  windowed_array = x[window_indexes]
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
  return windows, labels

def train_test_split_values(input_windows, targets, test_split = 0.25):
    split_point = int(len(input_windows)*(1- test_split))
    train_windows = input_windows[:split_point]
    train_labels = targets[:split_point]
    test_windows = input_windows[split_point:]
    test_labels = targets[split_point:]
    return train_windows, test_windows, train_labels, test_labels

def train_test_split_timestamps(timestamp_labels, test_split = 0.25):
    split_point = int(len(timestamp_labels)*(1 - test_split))
    train_timestamps = timestamp_labels[:split_point]
    test_timestamps = timestamp_labels[split_point:]
    return train_timestamps, test_timestamps
