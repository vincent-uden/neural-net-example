import numpy as np

from mnist import MNIST
from random import randint

def leaky_relu(z):
    if z > 0:
        return z
    else:
        return 0.01 * z

def leaky_relu_p(z):
    if z > 0:
        return 1
    else:
        return 0.01

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_p(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost(A, y):
    return (A - y) ** 2

def cost_p(A, y):
    return 2 * (A - y) 

def vectorize_label(i):
    output = np.zeros((1, 10))
    output[0][i] = 1
    return output

def parse_output(arr):
    return arr.argmax()

mn_data = MNIST('data-set')

images, labels = mn_data.load_training()

print(type(images))
print(type(labels))

def generate_batch(size):
    # Generates a batch of training data for 
    # stochastic gradient descent
    output_imgs = []
    output_lbls = []
    data_size = len(images)
    # Pick random training data
    for i in range(size):
        index = randint(0, data_size)
        output_imgs.append(images[index])
        output_lbls.append(labels[index])
    output_imgs = np.array(output_imgs)
    output_lbls = np.array(output_lbls)
    # Normalize the input array
    output_imgs = output_imgs / 255
    return output_imgs, output_lbls

layer_sizes = [784, 32, 16, 16, 10]

# Weights
w1 = np.random.random((layer_sizes[0], layer_sizes[1])) * 0.1 - 0.05
w2 = np.random.random((layer_sizes[1], layer_sizes[2])) * 0.1 - 0.05
w3 = np.random.random((layer_sizes[2], layer_sizes[3])) * 0.1 - 0.05
w4 = np.random.random((layer_sizes[3], layer_sizes[4])) * 0.1 - 0.05

# Biases
b1 = np.random.random((1, layer_sizes[1])) * 0.1 - 0.05
b2 = np.random.random((1, layer_sizes[2])) * 0.1 - 0.05
b3 = np.random.random((1, layer_sizes[3])) * 0.1 - 0.05
b4 = np.random.random((1, layer_sizes[4])) * 0.1 - 0.05

learning_rate = 0.1


for batch in range(100):
    X, y = generate_batch(100)

    for i in range(100):
        # Forward propagation
        z1 = images.dot(w1) + b1
        a1 = leaky_relu(z1)
        z2 = a1.dot(w2) + b2
        a2 = leaky_relu(z2)
        z3 = a2.dot(w3) + b3
        a3 = leaky_relu(z3)
        z4 = a2.dot(w3) + b3
        a4 = sigmoid(z3)

        # Back propagation
        error = cost(a4, labels)
        gradient = cost_p(a4, labels)
        e4 = gradient * sigmoid_p(z4)
        e3 = (e4.dot(w4.T)) * sigmoid_p(z3)
        e2 = (e3.dot(w3.T)) * sigmoid_p(z2)
        e1 = (e2.dot(w2.T)) * sigmoid_p(z1)

        b4 = b4 - (e4.mean(axis=0) * learning_rate)
        b3 = b3 - (e3.mean(axis=0) * learning_rate)
        b2 = b2 - (e2.mean(axis=0) * learning_rate)
        b1 = b1 - (e1.mean(axis=0) * learning_rate)

        w4 = w4 - ((a3.T.dot(e4)) * learning_rate)
        w3 = w3 - ((a2.T.dot(e3)) * learning_rate)
        w2 = w2 - ((a1.T.dot(e2)) * learning_rate)
        w1 = w1 - ((images.T.dot(e1)) * learning_rate)