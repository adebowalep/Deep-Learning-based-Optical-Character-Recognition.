<!DOCTYPE html>
<html>
<head>
    <title>Deep OCR README</title>
</head>
<body>
    <h1>Deep OCR</h1>
    <h2>Task Description</h2>
    <p>The task at hand is to train a deep learning model for Optical Character Recognition (OCR) to read characters from images. The objective is to develop an algorithm that can accurately recognize a fixed number of digits ranging from 0 to 9. The system will be provided with cropped license plates, each containing exactly 5 digits. For example, the algorithm should be able to read numbers like "37463" or "23837" accurately.</p>
    <p>To accomplish this, a complete example needs to be built, including the generation of artificial samples. The dataset should consist of at least 60,000 images, with corresponding labels indicating the correct digits. The model will be trained using the Keras library with TensorFlow 2.5.0 as the backend. The goal is to achieve the best possible performance in terms of accuracy and reliability for the OCR algorithm.</p>

<h2>Installation</h2>
    <p>Run the following command to install the required dependencies:</p>
    <pre><code>!pip install -r requirements.txt | grep -v "already satisfied"</code></pre>

 <h2>Running the Tests</h2>
    <p>Execute the following command to run the tests:</p>
    <pre><code>!pytest -vv src/helpers.py -k test_generate_images</code></pre>

<h2>Usage</h2>
    <p>Follow the code snippets below to use the deep OCR model:</p>

 <h3>Generating Training and Validation Images</h3>
    <pre><code>from src.helpers import generate_images

train_dataset, validation_dataset = generate_images(12000)

<h3>Augmenting the Data</h3>
    <pre><code>from src.helpers import augment_data
import tensorflow as tf

##### Apply data augmentation to the training dataset
num_augmentations = 5  # Number of augmented versions to generate for each sample
augmented_train_dataset = train_dataset.flat_map(lambda image, label: tf.data.Dataset.from_tensor_slices(augment_data(image, label, num_augmentations)))

##### Get the number of augmented samples
num_training_samples = augmented_train_dataset.reduce(0, lambda count, _: count + 1).numpy()
print("Number of augmented samples:", num_training_samples)</code></pre>

<h3>Training the Model</h3>
    <pre><code>from src.cnnmodel import create_rnn_model

model = create_rnn_model(input_shape, num_classes)

<h2>Model Architecture</h2>
<p>The model used for OCR is based on a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) layers. The architecture includes convolutional layers for feature extraction, followed by a flattening layer, a repeat vector layer, an LSTM layer, and finally time-distributed dense layers for classification.</p>

<h2>Checkpoints</h2>
    <p>The model weights are saved using checkpoints during the training process. The checkpoints are saved in the "checkpoints" directory. You can load the saved weights using the provided code.</p>

<h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
</html>
