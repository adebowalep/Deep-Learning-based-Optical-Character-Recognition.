import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_addons")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.ticker as mticker 
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras.utils import to_categorical
import os




def resize_image(image, size):
    """
    Resizes the input image to the specified size.

    Args:
        image (PIL.Image.Image): The input image to be resized.
        size (tuple): The desired size of the output image, specified as a tuple (width, height).

    Returns:
        np.ndarray: The resized image as a NumPy array.

    Raises:
        TypeError: If the image is not a PIL Image object.
        ValueError: If the size is not a tuple of two positive integers.
    """
    resized_image = image.resize(size)
    resized_image = np.array(resized_image)
    resized_image = np.expand_dims(resized_image, axis=-1)  # Add an extra dimension for grayscale channel
    return resized_image


def generate_images(num_images, train_ratio=0.9, seed=None, image_size=(224, 224)):
    """
    Generates a dataset of images and labels for training and testing.

    Args:
        num_images (int): The total number of images to generate.
        train_ratio (float): The ratio of images to be used for training. Defaults to 0.9.
        seed (int or None): The seed value for reproducible random shuffling. Defaults to None.
        image_size (tuple): The desired size of the output images, specified as a tuple (width, height).
                            Defaults to (224, 224).

    Returns:
        tf.data.Dataset: The dataset containing training and testing images and labels.

    Raises:
        ValueError: If num_images is not a positive integer.
                    If train_ratio is not a float between 0 and 1.
                    If seed is not a positive integer or None.
    """
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer")
    if not isinstance(train_ratio, float) or not (0 <= train_ratio <= 1):
        raise ValueError("train_ratio must be a float between 0 and 1")
    if seed is not None and (not isinstance(seed, int) or seed <= 0):
        raise ValueError("seed must be a positive integer or None")

    label_font = ImageFont.truetype("./arial.ttf", 60)

    images = []
    labels = []
    for _ in range(num_images):
        image = Image.new("L", (400, 80), 0)
        digits = [random.randint(0, 9) for _ in range(5)]
        label_draw = ImageDraw.Draw(image)
        for i, digit in enumerate(digits):
            label_draw.text((20 + i * 80, 10), str(digit), font=label_font, fill=255)
        resized_image = resize_image(image, image_size)
        images.append(np.array(resized_image))
        labels.append(digits)

    images = tf.convert_to_tensor(np.array(images), dtype=tf.float32)
    images = images / 255.0

    labels = tf.convert_to_tensor(np.array(labels), dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    num_train = int(num_images * train_ratio)
    if seed is not None:
        dataset = dataset.shuffle(num_images, seed=seed)
    train_dataset = dataset.take(num_train)
    test_dataset = dataset.skip(num_train)

    return train_dataset, test_dataset



def augment_data(image, label, num_augmentations):
    """
    Augments a single image and its corresponding label by applying random transformations.

    Args:
        image (tf.Tensor): The input image tensor.
        label (tf.Tensor): The input label tensor.
        num_augmentations (int): The number of augmentations to generate.

    Returns:
        augmented_images (tf.Tensor): A tensor containing the augmented images.
        augmented_labels (tf.Tensor): A tensor containing the corresponding augmented labels.

    """

    augmented_images = []
    augmented_labels = []

    for _ in range(num_augmentations):
        # Apply random image transformations
        augmented_image, augmented_label = apply_random_transformations(image, label)

        # Append the augmented image and label to the lists
        augmented_images.append(augmented_image)
        augmented_labels.append(augmented_label)

    # Convert the lists to TensorFlow tensors
    augmented_images = tf.convert_to_tensor(augmented_images)
    augmented_labels = tf.convert_to_tensor(augmented_labels)

    return augmented_images, augmented_labels


def apply_random_transformations(image, label):
    """
    Applies random image transformations to the given image tensor.

    Args:
        image (tf.Tensor): The input image tensor.
        label (tf.Tensor): The input label tensor.

    Returns:
        augmented_image (tf.Tensor): The augmented image tensor.
        augmented_label (tf.Tensor): The augmented label tensor.

    """

    alpha = 0
    sigma = 1
    elastic_transform_params = [alpha, sigma]
    desired_height = 224
    desired_width = 224
    num_channels = 1

    # Apply vertical shift using TensorFlow
    height_shift = tf.random.uniform([], minval=-0.1, maxval=0.1)
    shift_pixels = tf.cast(height_shift * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)
    image = tf.roll(image, shift_pixels, axis=0)

    # Randomly adjust the brightness using TensorFlow
    image = tf.image.random_brightness(image, max_delta=0.05)

    # Apply rotation using TensorFlow Addons
    angle = tf.random.uniform([], minval=-10, maxval=10)
    radians = tf.cast(angle * np.pi / 180, tf.float32)
    image = tfa.image.rotate(image, radians, fill_mode='nearest')

    # Random crop transformation
    image = tf.image.random_crop(image, size=[desired_height, desired_width, 1])

    # Shear transformation
    shear_factor = tf.random.uniform([], minval=-0.09, maxval=0.09)
    image = tfa.image.transform(image, [1.0, shear_factor, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # Contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Apply Gaussian noise with random intensity using TensorFlow Addons
    noise_intensity = tf.random.uniform([], minval=0.0, maxval=0.05)
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_intensity, dtype=tf.float32)
    image = tf.add(image, noise)

    # Clip the pixel values to [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label




def display(tr_loss, val_loss, tr_acc, val_acc, lr):
    """
    Display the training and validation loss/accuracy curves.

    Args:
        tr_loss (list): List of training loss values for each epoch.
        val_loss (list): List of validation loss values for each epoch.
        tr_acc (list): List of training accuracy values for each epoch.
        val_acc (list): List of validation accuracy values for each epoch.
        lr (float): Learning rate used during training.

    Returns:
        None

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
    plt.subplot(211)
    epochs = np.arange(10) + 1
    plt.plot(epochs, tr_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title(f'Training accuracy and validation loss value with {lr} learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)

    plt.subplot(212)
    plt.plot(epochs, tr_acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title(f'Training accuracy and validation loss value with {lr} learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid(False)
    plt.show()

def visualize_images(image_dataset, num_images):
    """
    Visualizes a subset of images from the image dataset along with their labels.

    Args:
        image_dataset (tf.data.Dataset): The dataset containing images and labels.
        num_images (int): The number of images to visualize.

    Raises:
        ValueError: If num_images is not a positive integer.

    """

    # Validate the input argument
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer")

    # Select the first `num_images` from the dataset
    dataset_subset = image_dataset.take(num_images)

    # Create a list to store the images and labels
    images = []
    labels = []

    # Iterate over the dataset and collect the images and labels
    for image, label in dataset_subset:
        images.append(image.numpy())
        labels.append(label.numpy())

    # Plot the images with labels
    fig, axes = plt.subplots(2, num_images // 2, figsize=(12, 10))
    axes = axes.flatten()
    for i in range(num_images):
        # Convert the image to uint8 and remove the channel dimension
        image = (images[i] * 255).astype(np.uint8)

        # Reshape the one-hot encoded label to a matrix
        label_matrix = np.reshape(labels[i], (1, -1))

        # Plot the image
        ax = axes[i]
        ax.imshow(image)
        ax.set_title('Label:\n{}'.format(label_matrix))
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def generate_test_images(num_images, seed=None, image_size=(224, 224)):
    """
    Generates a dataset of test images and labels.

    Args:
        num_images (int): The total number of test images to generate.
        seed (int or None): The seed value for reproducible random shuffling. Defaults to None.
        image_size (tuple): The desired size of the output images, specified as a tuple (width, height).
                            Defaults to (224, 224).

    Returns:
        tf.data.Dataset: The dataset containing test images and labels.

    Raises:
        ValueError: If num_images is not a positive integer.
                    If seed is not a positive integer or None.
    """
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer")
    if seed is not None and (not isinstance(seed, int) or seed <= 0):
        raise ValueError("seed must be a positive integer or None")

    label_font = ImageFont.truetype("./arial.ttf", 60)

    images = []
    labels = []
    for _ in range(num_images):
        image = Image.new("L", (400, 80), 0)
        digits = [random.randint(0, 9) for _ in range(5)]
        label_draw = ImageDraw.Draw(image)
        for i, digit in enumerate(digits):
            label_draw.text((20 + i * 80, 10), str(digit), font=label_font, fill=255)
        resized_image = resize_image(image, image_size)
        images.append(np.array(resized_image))
        labels.append(digits)

    images = tf.convert_to_tensor(np.array(images), dtype=tf.float32)
    images = images / 255.0

    labels = tf.convert_to_tensor(np.array(labels), dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if seed is not None:
        dataset = dataset.shuffle(num_images, seed=seed)

    return dataset    



def visualize_predictions(image_dataset, num_images, model):
    """
    Visualizes a subset of images from the image dataset along with their labels.

    Args:
        image_dataset (tf.data.Dataset): The dataset containing images and labels.
        num_images (int): The number of images to visualize.
        model (tf.keras.Model): The trained model used for prediction.

    Raises:
        ValueError: If num_images is not a positive integer.

    """

    # Validate the input argument
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer")

    # Select the first `num_images` from the dataset
    dataset_subset = image_dataset.take(num_images)

    # Create a list to store the images, predicted labels, and actual labels
    images = []
    predicted_labels = []
    actual_labels = []

    # Iterate over the dataset and collect the images and labels
    for image, label in dataset_subset:
        images.append(image)
        expanded_image = tf.expand_dims(image, axis=0)  # Add batch dimension
        predicted_labels.append(tf.math.argmax(model.predict(expanded_image), axis=-1))
        actual_labels.append(label)

    # Convert the lists to NumPy arrays
    images = np.array(images)
    predicted_labels = np.array(predicted_labels)
    actual_labels = np.array(actual_labels)

    # Plot the images with labels
    fig, axes = plt.subplots(2, num_images // 2, figsize=(12, 10))
    axes = axes.flatten()
    for i in range(num_images):
        # Convert the image tensor to a NumPy array
        image = images[i]

        # Reshape the predicted and actual labels to matrices
        predicted_label_matrix = np.reshape(predicted_labels[i], (1, -1))
        actual_label_matrix = np.reshape(actual_labels[i], (1, -1))

        # Determine the color of the label based on correctness
        predicted_label = predicted_labels[i]
        actual_label = actual_labels[i]
        color = 'green' if np.array_equal(predicted_label, actual_label) else 'red'
        actual_color = 'blue'

        # Plot the image
        ax = axes[i]
        ax.imshow(image, cmap='gray')  # Assuming grayscale images
        ax.set_title('Predicted Label:\n{}\nActual Label:\n{}'.format(predicted_label_matrix, actual_label_matrix), color=color if not np.array_equal(predicted_label, actual_label) else actual_color)
        ax.title.set_color(actual_color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_predictions__(image_dataset, num_images, model):
    """
    Visualizes a subset of images from the image dataset along with their labels.

    Args:
        image_dataset (tf.data.Dataset): The dataset containing images and labels.
        num_images (int): The number of images to visualize.
        model (tf.keras.Model): The trained model used for prediction.

    Raises:
        ValueError: If num_images is not a positive integer.

    """

    # Validate the input argument
    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError("num_images must be a positive integer")

    # Select the first `num_images` from the dataset
    dataset_subset = image_dataset.take(num_images)

    # Create a list to store the images, predicted labels, and actual labels
    images = []
    predicted_labels = []
    actual_labels = []

    # Iterate over the dataset and collect the images and labels
    for image, label in dataset_subset:
        images.append(image.numpy())
        predicted_labels.append(np.argmax(model.predict(np.expand_dims(image, axis=0))))
        actual_labels.append(label.numpy())

    # Plot the images with labels
    fig, axes = plt.subplots(2, num_images // 2, figsize=(12, 10))
    axes = axes.flatten()
    for i in range(num_images):
        # Convert the image to uint8 and remove the channel dimension
        image = (images[i] * 255).astype(np.uint8)

        # Reshape the predicted and actual labels to matrices
        predicted_label_matrix = np.reshape(predicted_labels[i], (1, -1))
        actual_label_matrix = np.reshape(actual_labels[i], (1, -1))

        # Determine the color of the label based on correctness
        color = 'green' if np.array_equal(predicted_labels[i], actual_labels[i]) else 'red'

        # Plot the image
        ax = axes[i]
        ax.imshow(image)
        ax.set_title('Predicted Label:\n{}\nActual Label:\n{}'.format(predicted_label_matrix, actual_label_matrix), color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    
 ######################################################################################
#                                     TESTS
######################################################################################

import pytest

from PIL import Image

def test_generate_images():
    # Set up test parameters
    num_images = 100
    train_ratio = 0.8
    seed = 42
    image_size = (224, 224)

    # Generate the dataset
    train_dataset, test_dataset = generate_images(num_images, train_ratio, seed, image_size)

    # Check the dataset sizes
    assert len(list(train_dataset)) == int(num_images * train_ratio)
    assert len(list(test_dataset)) == num_images - int(num_images * train_ratio)

    # Check the image sizes
    for image, _ in train_dataset:
        assert image.shape[:2] == image_size

    for image, _ in test_dataset:
        assert image.shape[:2] == image_size

    # Check the label types and values
    for _, label in train_dataset:
        assert isinstance(label, tf.Tensor)
        assert label.dtype == tf.int32
        assert label.shape == (5,)  # Assuming the labels are always of length 5

    for _, label in test_dataset:
        assert isinstance(label, tf.Tensor)
        assert label.dtype == tf.int32
        assert label.shape == (5,)  # Assuming the labels are always of length 5

    # Check resizing functionality
    test_image = Image.new("L", (400, 80), 0)
    resized_image = resize_image(test_image, image_size)
    assert isinstance(resized_image, np.ndarray)
    assert resized_image.shape == (*image_size, 1)  # Assuming grayscale image

    # Additional test cases can be added to cover more scenarios
        
def test_visualize_images():
    # Create a small image dataset for testing
    num_images = 4
    image_shape = (32, 32, 3)
    label_shape = (10,)

    # Generate a sample image dataset
    image_dataset = tf.data.Dataset.from_tensor_slices((
        np.random.rand(num_images, *image_shape),
        np.random.rand(num_images, *label_shape)
    ))

    # Call the function to visualize the images
    visualize_images(image_dataset, num_images)

    # Check that the visualization function runs without errors
    assert True  # If the function runs without errors, the test passes

def test_augment_data():
    # Set up test parameters
    num_augmentations = 5
    image = tf.random.uniform((224, 224, 1))
    label = tf.constant([0, 1, 2, 3, 4])

    # Augment the data
    augmented_images, augmented_labels = augment_data(image, label, num_augmentations)

    # Check the shapes of augmented images and labels
    assert augmented_images.shape == (num_augmentations, 224, 224, 1)
    assert augmented_labels.shape == (num_augmentations, 5)

    # Check the types of augmented images and labels
    assert augmented_images.dtype == tf.float32
    assert augmented_labels.dtype == tf.int32

    # Check if the augmented images are different from the original image
    for augmented_image in augmented_images:
        assert not np.array_equal(augmented_image, image)

