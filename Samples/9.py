from numpy import exp, array, random, dot


# нейронка на 3 входных и один выходной нейрон
class NeuralNetwork:
    def __init__(self):
        random.seed(1)  # задаем генератору случайных числе стандартное значение
        self.synaptic_weights = 2 * random.random((3, 1)) - 1  # нейронная сеть (3 x 1) -1 для 0-1 диапазона

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output  # высчитываем ошибку на основе реальных полученных значений и обучения

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))  # выведем результат нашей нейронной сети


if __name__ == "__main__":
    neural_network = NeuralNetwork()  # инициализируем нейронную сеть

    print("Веса нейронной сети: ")
    print(neural_network.synaptic_weights, '\n')

    # обучающая выборка (входные и выходные нейроны)
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 100000)

    print("Новые веса после тренировки: ")
    print(neural_network.synaptic_weights, '\n')

    # Test the neural network with a new situation.
    print("Предскажем новую ситуацию [1, 0, 0] -> ?: ", neural_network.think(array([1, 0, 0])))
