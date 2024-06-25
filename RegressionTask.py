from seed import Seed
import numpy as np

import matplotlib.pyplot as plt
import time

from NeuralNetwork import NeuralNetwork

np.random.seed(Seed().getSeed())


# function fo be approximated
def f1(x):
    return np.log10(x + 1) * np.sin(3 * x + 1)
    # return np.sin(3 * x)


def regression_task1(shape, activation_function='tanh',
                     Ng=10, Nf=100, dT=1.0, lr=0.5,
                     maxiter=10000, plots=True,
                     optimizer='ParaFlowS'):
    N = 20  # number of training points
    x_train = np.linspace(0, np.pi, N + 1)  # training points
    y_train = f1(x_train)  # training values

    # maxiter = int(1.5e4) #6e4) #1.5e4)
    # lr = 5e-3
    # lr = 0.5
    # lr = 0.75

    NN = NeuralNetwork.NeuralNetwork(shape)

    start_time = time.time()
    if optimizer == 'GD':
        NN.trainGD(x_train, y_train, maxiter, lr, activation_function)
    elif optimizer == 'Adam':
        NN.trainAdam(x_train, y_train, maxiter, lr, activation_function)
    elif optimizer == 'ParAdamFlowS':
        NN.trainParAdamFlowS(x_train, y_train, Ng=Ng, Nf=Nf, dT=dT, dt=lr, maxiter=maxiter, lr=lr,
                             activation_function=activation_function)
    elif optimizer == 'ParaFlowS':
        NN.trainParaFlowS(x_train, y_train, Ng=Ng, Nf=Nf, dT=dT, dt=lr, maxiter=maxiter, lr=lr,
                          activation_function=activation_function)
    elif optimizer == 'Linear':
        NN.trainLinear(x_train, y_train, activation_function)
    else:
        print('Optimizer not recognized!')
        return

    # NN.trainGD(x_train, y_train, maxiter, lr, activation_function)
    # NN.trainLinear(x_train, y_train, activation_function)

    # NN.trainParareal(x_train, y_train, maxiter, lr, activation_function, Ncoarse=6, Nparareal=4)
    # NN.trainParaFlowS2(x_train, y_train, Ng=param, Nf=100, dT=3.0, dt=lr, maxiter=maxiter, lr=lr, activation_function=activation_function)
    # NN.trainParAdamFlowS(x_train, y_train, Ng=param, Nf=50, dT=0.9, dt=lr, maxiter=maxiter, lr=lr, activation_function=activation_function) #, restart = restart)
    end_time = time.time()
    print('training time:', end_time - start_time)

    if plots:
        # plot the cost history
        plt.semilogy(NN.costHistory[0:129012]) #, marker='.')
        # plt.xlabel('k (outer iterations)')
        plt.xlabel('GD iterations')
        # plt.xlabel('Adam iterations')
        plt.ylabel('cost')
        plt.title('Cost history')
        plt.show()

    print('minimum cost:', np.min(NN.costHistory))
    print('\n\n')

    # create some test points
    if plots:
        x_test = np.linspace(0, np.pi, 5 * N)
        y_pred = NN.predictRegression(x_test)

        # plots
        plt.plot(x_test, f1(x_test), label='true')
        plt.plot(x_test, y_pred, label='predicted')
        plt.legend()
        plt.show()
