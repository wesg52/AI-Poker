import random
import numpy as np

def relu(input, leak=0):
    return max(input, input*leak)

class Layer(object):
    def __init__(self, num_nuerons, num_inputs):
        self.weights = np.random.normal(size=(num_nuerons, num_inputs+1))

    def output(input):
        return np.matmul(self.weights, input)


features = np.array([1,2,3,4])
l1 = Layer(3, 4)
l2 = Layer(2, 3)

output = relu(l2.output(relu(l1.output(features))))
