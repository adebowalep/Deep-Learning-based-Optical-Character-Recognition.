import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers


def create_rnn_model(input_shape, num_classes):
    """
    Creates a recurrent neural network (RNN) model for sequence generation.

    The model architecture consists of a convolutional layer followed by a LSTM layer
    and time-distributed dense layers to generate a sequence of labels.

    Args:
        input_shape (tuple): Shape of the input image tensor, e.g., (height, width, channels).
        num_classes (int): Number of classes (labels) for sequence generation.

    Returns:
        tf.keras.Model: RNN model for sequence generation.

    Raises:
        ValueError: If input_shape is not a tuple or num_classes is not a positive integer.

    """

    # Validate the input arguments
    if not isinstance(input_shape, tuple):
        raise ValueError("input_shape must be a tuple")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    # Create the RNN model
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.RepeatVector(5),  # Repeat the flattened vector for 5 time steps
        layers.LSTM(64, return_sequences=True),  # LSTM layer for sequence generation
        layers.TimeDistributed(layers.Dense(64, activation='relu')),  # Apply Dense layer to each time step
        layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))  # Apply Dense layer to each time step
    ])

    return model

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.mark.parametrize(
    "input_shape, num_classes",
    [
        ((224, 224, 3), 10),  # Example 1
        ((128, 128, 1), 5),   # Example 2
        ((300, 300, 3), 20)   # Example 3
    ]
)
def test_create_rnn_model(input_shape, num_classes):
    # Call the function to create the model
    model = create_rnn_model(input_shape, num_classes)

    # Check if the returned value is an instance of tf.keras.Model
    assert isinstance(model, tf.keras.Model)

    # Check if the model has the expected number of layers
    assert len(model.layers) == 7

    # Check if the input shape of the first layer matches the specified input shape
    assert model.layers[0].input_shape[1:] == input_shape
