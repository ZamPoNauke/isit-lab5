from numpy import exp, array, random, dot
import os


def image_to_data(image_file):  # принимаем файл и преобразуем его значения в 0 и 1
    import cv2

    img = cv2.imread(image_file)
    data = []

    for i in img:
        for j in i:
            img_list = j.tolist()
            if int(img_list[0]) == 0:
                data.append(1)
            else:
                data.append(0)
    return data


class NeuralNetwork:
    def __init__(self, name):
        self.debug = False
        random.seed(1)  # задаем генератору случайных числе стандартное значение
        self.synaptic_weights = 2 * random.random((1024, 1)) - 1  # нейронная сеть (1024 x 1) -1 для 0-1 диапазона
        self.name = name

    def __sigmoid(self, x):
        return 1 / (1 + exp(-2 * x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            # Высчитываем ошибку на основе реальных полученных значений и обучения
            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Изменим веса
            # print('Веса', self.synaptic_weights, '\n', adjustment)
            self.synaptic_weights += adjustment

    def think(self, inputs):
        if self.debug == True:
            print('\ninputs:\n', inputs)
            print(' веса:', self.synaptic_weights.T)
            print(' результат:', dot(inputs, self.synaptic_weights).T)  # на основе весов и входных данных
            print(' результат после сигмоида:', self.__sigmoid(dot(inputs, self.synaptic_weights)).T)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))  # выведем результат нашей нейронной сети

    def weight_print(self):
        print("Веса нейронной сети", self.name + ": ")
        print(self.synaptic_weights, '\n')


def learning(NN_name, folder):
    # обучающая выборка (входные и выходные нейроны)
    # NN_name.weight_print()
    folder_line = os.path.join(os.getcwd(), folder)

    training_set_inputs_data, training_set_output_data = [], []

    for folder in os.listdir(folder_line):
        fullname = os.path.join(folder_line, str(folder))
        if os.path.isfile(fullname):
            # print(fullname)
            training_set_inputs_data.append(image_to_data(fullname))  # добавляем данные всех файлов в массив
            training_set_output_data.append(1)

    folder_line = os.path.join(os.getcwd(), 'images\\line\\')
    for folder in os.listdir(folder_line):
        fullname = os.path.join(folder_line, str(folder))
        if os.path.isfile(fullname):
            # print(fullname)
            training_set_inputs_data.append(image_to_data(fullname))  # добавляем данные всех файлов в массив
            training_set_output_data.append(0)


    training_set_inputs = array(training_set_inputs_data)
    training_set_outputs = array([training_set_output_data]).T

    NN_name.train(training_set_inputs, training_set_outputs, 500)  # обучаем нейроную сеть на наших данных
    # NN_name.weight_print()

if __name__ == "__main__":
    # инициализируем нейронную сеть
    neural_network_tang = NeuralNetwork('Треугольник')
    neural_network_plus = NeuralNetwork('Крест')
    neural_network_squa = NeuralNetwork('Квадрат')

    whole_network = [neural_network_tang, neural_network_plus, neural_network_squa]

    learning(neural_network_tang, 'images\\tangle\\')
    learning(neural_network_plus, 'images\\plus\\')
    learning(neural_network_squa, 'images\\square\\')

    best_result = 0.0
    best_result_name = ''
    image = ('D:\\Development\\GitHub\\isit-lab5\\images\\test.png')
    print("\nПредскажем новую ситуацию на основе изображения -> ?: ")
    for perceptron in whole_network:
        if float(perceptron.think(image_to_data(image))) > best_result:
            best_result = float(perceptron.think(image_to_data(image))[0])
            best_result_name = perceptron.name
    print('Ответ сети: ', best_result_name, best_result)
