from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt


# Select only two digits (e.g., 0 and 1)
DIGIT_1 = 2  # First digit to classify
DIGIT_2 = 5  # Second digit to classify


def binary_cross_entropy(y, yHat):
    epsilon = 1e-15
    yHat = tf.clip_by_value(yHat, epsilon, 1-epsilon)
    return -tf.reduce_mean(y * tf.math.log(yHat) + (1-y) * tf.math.log(1-yHat))


if __name__ == "__main__":
    # 載入MNIST資料集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Create masks for the selected digits
    train_mask = (train_labels == DIGIT_1) | (train_labels == DIGIT_2)
    test_mask = (test_labels == DIGIT_1) | (test_labels == DIGIT_2)

    # Filter the dataset
    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask]
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]

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
    network.add(Dense(1, activation='sigmoid'))
    network.compile(optimizer='rmsprop', loss=binary_cross_entropy, metrics=['accuracy'])
    print(network.summary)

    # 訓練階段
    network.fit(train_images, train_labels, epochs=10, batch_size=200)
    # 測試階段
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print("Test Accuracy:", test_acc)
