class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.prime_loss = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, prime_loss):
        self.loss = loss
        self.prime_loss = prime_loss

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(output, y_train[j])

                error = self.prime_loss(output, y_train[j])
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print(f"Epoch:{i}, Success:{(1-err)*100}")
