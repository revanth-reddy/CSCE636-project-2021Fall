import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    x_values = []
    y_values = []
    for file in os.listdir(data_dir):
        if "data_batch" in file:
            filename = os.path.join(data_dir, file)
            #print(filename)
            with open(filename, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
#                 for key in dict:
#                     print (key)
                x_values.append(dict[b'data'])
                labels = np.asarray(dict[b'labels'])
                y_values.append(labels)
#                 print ("dict['data']: ", dict[b'data'].shape)
#                 print ("dict['labels']: ", labels.shape)
        
    x_train = np.concatenate(x_values, dtype=np.float32)
    y_train = np.concatenate(y_values, dtype=np.int32)
    print ("x_shape", x_train.shape)
    print ("y_shape", y_train.shape)
    x_values=[]
    y_values=[]
    for file in os.listdir(data_dir):
        if "test_batch" in file:
            filename = os.path.join(data_dir, file)
            with open(filename, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                x_values.append(dict[b'data'])
                labels = np.asarray(dict[b'labels'])
                y_values.append(labels)
    x_test = np.concatenate(x_values, dtype=np.float32)
    y_test = np.concatenate(y_values, dtype=np.int32)
    print ("x_test_shape", x_test.shape)
    print ("y_test_shape", y_test.shape)
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.9):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index = (int)(y_train.shape[0]*train_ratio)//1
    # print(split_index)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

