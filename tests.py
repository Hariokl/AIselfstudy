import json

from funcs import *

from traying_to_make_my_own import *
# from teset_neural import *
from keras.datasets import mnist
from keras.utils import np_utils


def mnst():
    global x_train, y_train, x_test, y_test
    # MNIST from server
    data = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    # training data : 60000 samples
    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)

    # same for test data : 10000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_test, y_test = unison_shuffled_copies(x_test, y_test)


def main(*, train, test, read_from_file, read_file, read_from, write_to_file, write_file, write_to, i):
    net = Network()

    if read_from_file:
        with open(read_file, "r") as file:
            data = json.load(file)
            data_net = data[read_from]
            for weights, biases in data_net:
                net.add(Layer(len(weights), len(weights[0]), np.array(weights), np.array(biases)))
    else:
        sizes = [(784, 50), (50, 20), (20, 10)]
        for size in sizes:
            net.add(Layer(*size, init_weights(*size), init_biases(1, size[1])))

    if train:
        net.fit(x_train=x_train[0:1000], y_train=y_train[0:1000], epochs=1000, learning_rate=0.01, draw_map=False)
        if write_to_file:
            with open(write_file, "r+") as jsf:
                data = dict()
                data[write_to] = net.get_data()
                jsf.seek(0)
                json.dump(data, jsf, ensure_ascii=False)
                jsf.truncate()

    if test:
        net.predict(x_test[i*1000:(i+1)*1000], y_test[i*1000:(i+1)*1000])


if __name__ == "__main__":
    params = {
        "read_from_file": True,
        "write_to_file": True,
        "write_to": "1",
        "read_from": "1",
        "read_file": "data1.json",
        "write_file": "data1.json",
        "train": False,
        "test": True
    }

    for i in range(100):
        mnst()
        print(f"\nALIVE FOR {i+1}")
        main(**params, i=0)
