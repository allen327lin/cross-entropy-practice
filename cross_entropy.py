from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time, localtime, strftime
import os
from pathlib import Path


CHARTS_FOLDER = "./charts/"


def cross_entropy(y, yHat):
    """
    Calculates the cross-entropy loss between true labels and predicted probabilities
    using TensorFlow operations.
    
    Parameters:
    y: Ground truth labels, one-hot encoded (tf.Tensor)
    yHat: Predicted probabilities from the model (tf.Tensor)
    
    Returns:
    tf.Tensor: The cross-entropy loss value
    """
    # 設定數值邊界
    epsilon = 1e-15
    # 為了避免 yHat 跑到無窮大或無窮小，因此設定 yHat 邊界
    yHat = tf.clip_by_value(yHat, epsilon, 1-epsilon)
    # 回傳 Cross Entropy 的 Loss
    return -tf.reduce_mean(tf.reduce_sum(y * tf.math.log(yHat), axis=-1))

def plot_training_history(history):
    """
    Plot training & validation accuracy and loss values.
    
    Parameters:
    history: Keras history object from model.fit()
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()

    # Save the plot
    format_time = strftime('%Y-%m-%d_%H-%M-%S', localtime(time()))
    chart_filename_stem = CHARTS_FOLDER + "Cross-Entropy_" + format_time
    if not Path(CHARTS_FOLDER).exists():
        os.mkdir(CHARTS_FOLDER)
    plt.savefig(chart_filename_stem, dpi=300, bbox_inches='tight')
    print("\n", f"Chart saved as {chart_filename_stem}.png")

    # Display the plot
    # plt.show()


if __name__ == "__main__":
    # 載入MNIST資料集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 資料前處理
    train_images = train_images.reshape((train_images.shape[0], 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((test_images.shape[0], 28 * 28))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # 建立人工神經網路
    network = Sequential()
    network.add(Dense(512, activation='relu', input_shape=(784,)))
    network.add(Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss=cross_entropy, metrics=['accuracy'])
    print(network.summary)

    # 訓練階段
    history = network.fit(train_images, train_labels, epochs=10, batch_size=200, validation_split=0.2)
    plot_training_history(history)
    # 測試階段
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print("Test Accuracy:", test_acc)
