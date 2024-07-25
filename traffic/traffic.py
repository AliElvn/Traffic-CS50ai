import argparse
import logging
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Train a traffic sign classification model.')
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('model_file', type=str, nargs='?', help='Optional model file to save')
    args = parser.parse_args()

    if len(sys.argv) not in [2, 3]:
        logging.error("Usage: python traffic.py data_directory [model.h5]")
        sys.exit(1)

    if not os.path.exists(args.data_directory):
        logging.error("Error: Data directory does not exist or is inaccessible.")
        sys.exit(1)

    try:
        images, labels = load_data(args.data_directory)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

    try:
        labels = tf.keras.utils.to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        sys.exit(1)

    np.random.seed(42)
    tf.random.set_seed(42)

    model = get_model()

    try:
        model.fit(x_train, y_train, epochs=EPOCHS)
        model.evaluate(x_test,  y_test, verbose=2)
    except Exception as e:
        logging.error(f"Error in model training or evaluation: {e}")
        sys.exit(1)

    if args.model_file:
        model.save(args.model_file)
        logging.info(f"Model saved to {args.model_file}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    print(f'Loading images from dataset in directory "{data_dir}"')

    images = []
    labels = []

    # Iterate through sign folders in directory:
    for foldername in os.listdir(data_dir):
        # Error Checking Data Folder
        try:
            int(foldername)
        except ValueError:
            print("Warning! Non-integer folder name in data directory! Skipping...")
            continue
    # Iterate through images in each folder
        for filename in os.listdir(os.path.join(data_dir, foldername)):
            # Open each image and resize to be IMG_WIDTH X IMG HEIGHT
            img = cv2.imread(os.path.join(data_dir, foldername, filename))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Normalise image pixel intensities:
            img=img/255

            # Append Resized Image and its label to lists
            images.append(img)
            labels.append(int(foldername))

    # Check number of Images Matches Number of Labels:
    if len(images) != len(labels):
        sys.exit('Error when loading data, number of images did not match number of labels!')
    else:
        print(f'{len(images)}, {len(labels)} labelled images loaded successfully from dataset!')

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Create the Neural Network Model using keras:
    model = tf.keras.models.Sequential([

    # Add 2 sequential 64 filter, 3x3 Convolutional Layers Followed by 2x2 Pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten layers
    tf.keras.layers.Flatten(),

    # Add A Dense Hidden layer with 512 units and 50% dropout
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add Dense Output layer with 43 output units
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Set additional model settings and compile:
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # Return model for training and testing
    return model


if __name__ == "__main__":
    main()
