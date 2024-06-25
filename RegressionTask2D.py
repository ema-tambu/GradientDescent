from seed import Seed
import numpy as np

import matplotlib.pyplot as plt
import time

from NeuralNetwork import NeuralNetwork

np.random.seed(Seed().getSeed())


# function fo be approximated
def f(x):
    return np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1])


def regression_task2D(shape, activation_function='tanh'):
    N = 20 # number of training points
    x1 = np.linspace(0, np.pi, N + 1)
    x2 = np.linspace(0, np.pi, N + 1)
    tmp1, tmp2 = np.meshgrid(x1, x2)
    x_train = np.array([tmp1.flatten(), tmp2.flatten()]).T
    y_train = f(x_train)

    maxiter = int(2e5)# 6e4)
    # lr = 5e-3
    lr = 0.75

    NN = NeuralNetwork.NeuralNetwork(shape)

    start_time = time.time()
    NN.trainGD(x_train, y_train, maxiter, lr, activation_function)
    # NN.trainParareal(x_train, y_train, maxiter, lr, activation_function, Ncoarse=6, Nparareal=4)
    # NN.trainLinear(x_train, y_train, activation_function)
    # NN.trainParaFlowS(x_train, y_train, Ng=6, Nf=20, dT=3, dt=lr, maxiter=maxiter, lr=lr, activation_function=activation_function)
    end_time = time.time()
    print('training time:', end_time - start_time)

    # plot the cost history
    plt.semilogy(NN.costHistory)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Cost history')
    plt.show()

    # print the final cost
    try:
        print('final cost:', NN.costHistory[-1])
    except:
        pass
    # print('cost history:', NN.costHistory)

    # create some test points for 2D regression testing
    x_test1 = np.linspace(0, np.pi, 5 * N)
    x_test2 = np.linspace(0, np.pi, 5 * N)
    tmp1, tmp2 = np.meshgrid(x_test1, x_test2)
    x_test = np.array([tmp1.flatten(), tmp2.flatten()]).T

    y_pred = NN.predictRegression(x_test)

    # plots for 2D regression
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_test[:, 0], x_test[:, 1], f(x_test), label='true')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_test[:, 0], x_test[:, 1], y_pred, label='predicted')
    plt.legend()
    plt.show()

