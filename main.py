import tensorflow as tf

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    print(mnist.size())
    print("main")