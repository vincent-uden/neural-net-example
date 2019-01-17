import numpy as np
import matplotlib.pyplot as plt

from mnist import MNIST

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

training_data = MNIST('data-set')

data = training_data.load_training_in_batches(5)

images, labels = data.__next__()
images = np.array(images)
images = images / 255
new_labels = []
for label in labels:
    new_labels.append(vectorize_label(label))
labels = np.vstack(new_labels)

# Images are 28 * 28         # There are 10 digits
layer_sizes = [28 * 28, 16, 16, 10]

w1 = np.random.random((layer_sizes[0], layer_sizes[1])) * 0.1 - 0.05
w2 = np.random.random((layer_sizes[1], layer_sizes[2])) * 0.1 - 0.05
w3 = np.random.random((layer_sizes[2], layer_sizes[3])) * 0.1 - 0.05

b1 = np.random.random((1, layer_sizes[1])) * 0.1 - 0.05
b2 = np.random.random((1, layer_sizes[2])) * 0.1 - 0.05
b3 = np.random.random((1, layer_sizes[3])) * 0.1 - 0.05

learning_rate = 0.1

print("Training...")
np.set_printoptions(threshold=np.nan)
for i in range(10000):
    z1 = images.dot(w1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(w3) + b3
    a3 = sigmoid(z3)

    error = cost(a3, labels)
    gradient = cost_p(a3, labels)
    e3 = gradient * sigmoid_p(z3)
    e2 = (e3.dot(w3.T)) * sigmoid_p(z2)
    e1 = (e2.dot(w2.T)) * sigmoid_p(z1)

    b3 = b3 - (e3.mean(axis=0) * learning_rate)
    b2 = b2 - (e2.mean(axis=0) * learning_rate)
    b1 = b1 - (e1.mean(axis=0) * learning_rate)

    w3 = w3 - ((a2.T.dot(e3)) * learning_rate)
    w2 = w2 - ((a1.T.dot(e2)) * learning_rate)
    w1 = w1 - ((images.T.dot(e1)) * learning_rate)

X = images
y = labels
# for i in range(10000):
#     z1 = X.dot(w1) + b1
#     a1 = sigmoid(z1)
#     z2 = a1.dot(w2) + b2
#     a2 = sigmoid(z2)
#     z3 = a2.dot(w3) + b3
#     a3 = sigmoid(z3)

#     # Back propagation
#     C = cost(a3, y)
#     d_C = cost_p(a3, y)
#     e3 = d_C * sigmoid_p(z3)
#     e2 = (e3.dot(w3.T)) * sigmoid_p(z2)
#     e1 = (e2.dot(w2.T)) * sigmoid_p(z1)

#     b3 = b3 - e3.mean(axis=0) * learning_rate
#     b2 = b2 - e2.mean(axis=0) * learning_rate
#     b1 = b1 - e1.mean(axis=0) * learning_rate

#     w3 = w3 - (a2.T.dot(e3)) * learning_rate
#     w2 = w2 - (a1.T.dot(e2)) * learning_rate
#     w1 = w1 - (X.T.dot(e1))  * learning_rate

print(f"y: {labels}")
print(f"a: {a3}")
#print(f"e: {C}")
print(b1.shape, b2.shape, b3.shape)
print(w1.shape, w2.shape, w3.shape)


#print("IMAGES")
#print(images)

#print("LABELS")
#print(labels)

#plt.matshow(images[0].reshape(28, 28), cmap='gray')
#plt.show()
