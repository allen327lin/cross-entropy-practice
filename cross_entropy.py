from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt


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
    # Add small epsilon to prevent taking log of zero
    epsilon = 1e-15
    # Clip predictions to prevent log(0)
    yHat = tf.clip_by_value(yHat, epsilon, 1-epsilon)
    
    # Calculate cross entropy
    # For each sample, sum -y * log(yHat) across all classes
    return -tf.reduce_mean(tf.reduce_sum(y * tf.math.log(yHat), axis=-1))


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
    network.fit(train_images, train_labels, epochs=10, batch_size=200)
    # 測試階段
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print("Test Accuracy:", test_acc)
