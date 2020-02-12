from keras.datasets import cifar10
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from joblib import load, dump

from keras.models import Sequential
from keras.layers.core import Dense, Activation

import matplotlib.pyplot as plt

import random
random.seed(159, 2)
validation = range(0, 10000)
sample_fashion = random.sample(range(0, 60000), 10000)
sample_cifar = random.sample(range(0, 50000), 10000)

(x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()
(x_train_c, y_train_c), (x_test_c, y_test_c) = cifar10.load_data()

x_train_f = [i for i in x_train_f]
y_train_f = [i for i in y_train_f]

x_train_c = [x_train_c[i] for i in sample_cifar]
y_train_c = [y_train_c[i] for i in sample_cifar]

x_test_f = [x_test_f[i] for i in validation]
y_test_f = [y_test_f[i] for i in validation]
x_test_c = [x_test_c[i] for i in validation]
y_test_c = [-1 for i in validation]

for i in range(len(x_train_f)):
    im = x_train_f[i]
    x_train_f[i] = cv2.resize(im, (32, 32))

for i in range(len(x_test_f)):
    im = x_test_f[i]
    x_test_f[i] = cv2.resize(im, (32, 32))

for i in range(len(x_train_c)):
    im = x_train_c[i]
    x_train_c[i] = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

for i in range(len(x_test_c)):
    im = x_test_c[i]
    x_test_c[i] = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


nbins = 9  # broj binova
cell_size = (4, 4)  # broj piksela po celiji
block_size = (4, 4)  # broj celija po bloku

hog = cv2.HOGDescriptor((32, 32),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

x_train_f = [hog.compute(img) for img in x_train_f]
x_train_c = [hog.compute(img) for img in x_train_c]

x_train = [x_train_f[i] for i in sample_fashion]+x_train_c

y_train = np.ones((1, 10000)).tolist()[0]+np.zeros((1, 10000)).tolist()[0]
y_train = np.array(y_train)

x_train = np.array(x_train)
x, y, z = x_train.shape
x_train = x_train.reshape(x, y*z)


svm = SVC(gamma='scale', degree=3, kernel='poly')
try:
    svm = load('./svm.sm')
except:
    svm.fit(x_train, y_train)
    dump(svm, 'svm.sm')

net = Sequential()

net.add(Dense(1200, input_dim=3600, activation='relu'))
net.add(Dense(400, activation='relu'))
net.add(Dense(10, activation='softmax'))

x_train_f = np.array(x_train_f)
x, y, z = x_train_f.shape
x_train_f = x_train_f.reshape(x, y*z)
try:
    net.load_weights('./weights.h5')
except:
    net.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
    net.fit(x_train_f, np.array(y_train_f),
            batch_size=3000, epochs=83, verbose=1)

    net.save_weights('./weights.h5')

x_test_f = [hog.compute(img) for img in x_test_f]
x_test_c = [hog.compute(img) for img in x_test_c]

x_test = x_test_c+x_test_f

x_test = np.array(x_test)
x, y, z = x_test.shape
x_test = x_test.reshape(x, y*z)

y_test = y_test_c+y_test_f

results = []

for img in x_test:
    img = np.array([img])
    is_cloth = svm.predict(img)[0]
    if is_cloth == 0:
        results.append(-1)
    else:
        t = net.predict(img)[0].argmax()
        results.append(t)

print("Tacnost: ", accuracy_score(np.array(y_test), np.array(results)))
