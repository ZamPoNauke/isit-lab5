import numpy as np

# Функция актицации
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


# https://gist.github.com/maunashjani/922e5a2a60130367dc58f1ee1fd6da36
# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]), dtype=float)
y = np.array(([1], [0], [1], [1]), dtype=float)

NN = NeuralNetwork(X, y)
for i in range(1000):  # trains the NN 1,000 times
    print("for iteration # " + str(i) + "\n")
    print("Input : \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(NN.feedforward()))
    print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
    print("\n")

NN.train(X, y)