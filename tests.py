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


def main(read_from_file=False, write_to_file=True, train=True, test=False):
    if read_from_file:
        with open("cursed.json", "r") as file:
            data = json.load(file)
            data_net = data["1"]
            layers = []
            for weights, biases in data_net:
                layers.append(Layer(len(weights), len(weights[0]), weights=np.array(weights), biases=np.array(biases)))
    else:
        sizes = [(784, 50), (50, 20), (20, 10)]
        layers = [Layer(*x) for x in sizes]
    net = Network(layers)
    if train:
        net.fit(x_train=x_train[0:1000], y_train=y_train[0:1000], epochs=1000, learning_rate=0.3, draw_map=True)
        if write_to_file:
            with open("cursed.json", "r+") as jsf:
                data = dict()
                data["1"] = net.get_data()
                jsf.seek(0)
                json.dump(data, jsf, ensure_ascii=False)
                jsf.truncate()
    if test:
        net.predict(x_test[:1000], y_test[:1000])


if __name__ == "__main__":
    for i in range(100):
        mnst()
        print(f"\nALIVE FOR {i+1}")
        main(read_from_file=True, write_to_file=True, train=True, test=True)
