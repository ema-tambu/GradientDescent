from seed import Seed
import numpy as np

np.random.seed(Seed().getSeed())

from Optimizers.GD import GradientDescent
from Optimizers.Adam import Adam
from Optimizers.Parareal import Parareal
from Optimizers.ParaFlowS import ParaFlowS
from Optimizers.ParaFlowS2 import ParaFlowS2
from Optimizers.ParAdamFlowS import ParAdamFlowS
from Optimizers.Linear import Linear

class NeuralNetwork:
    def __init__(self, shape):
        self.shape = shape
        self.L = len(self.shape)
        self.shapes = []  # useful because we store all the parameters in a single vector
        for i in range(self.L - 1):
            self.shapes.append(np.array([self.shape[i + 1], self.shape[i]]))

        self.maxiter = int(1e3)
        self.lr = 1e-3
        self.activation_function = 'sigmoid'

        self.costHistory = None
        self.p = 0.5 * np.random.normal(0, 1, (
        np.sum([np.prod(s) for s in self.shapes]) + np.sum([s[0] for s in self.shapes]),))
        # self.p = np.ones(np.sum([np.prod(s) for s in self.shapes]) + np.sum([s[0] for s in self.shapes]))
        self.x_train = None
        self.y_train = None
        self.batchSize = None

        # initialize stuff for network differentiation
        self.dLp_ = np.zeros(self.p.shape)
        self.z = None  # [np.ones((self.shape[i + 1], self.batchSize)) for i in range(self.L - 1)]
        self.delta = None  # [np.ones((self.shape[i + 1], self.batchSize)) for i in range(self.L - 1)]
        self.a = None  # [np.ones((self.shape[i], self.batchSize)) for i in range(self.L)]

    def sigma(self, x):
        if self.activation_function == 'sigmoid':
            print('using sigmoid')
            return 1 / (1 + np.exp(-x))
        if self.activation_function == 'tanh':
            return np.tanh(x)
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        if self.activation_function == 'leaky_relu':
            return np.maximum(0.01 * x, x)

    def sigmaPrime(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigma(x) * (1 - self.sigma(x))
        if self.activation_function == 'tanh':
            return 1 - self.sigma(x) ** 2
        if self.activation_function == 'relu':
            return (x > 0).astype(int)
        if self.activation_function == 'leaky_relu':
            return (x > 0).astype(int) + 0.01 * (x <= 0).astype(int)

    # _____ functions for network training _____

    def trainGD(self, x_train, y_train, maxiter=int(1e3), lr=1e-3, activation_function='sigmoid'):
        self.x_train = x_train
        self.y_train = y_train
        # self.batchSize = x_train.shape[-1]  # for now only full gradient descent
        self.batchSize = x_train.shape[0]

        self.activation_function = activation_function
        self.maxiter = maxiter
        self.lr = lr

        self.init_training()  # initialize a, z, delta now that we know the batch size !!!

        optimizer = GradientDescent(self)
        self.p, self.costHistory = optimizer.train()

    def trainAdam(self, x_train, y_train, maxiter=int(1e3), lr=1e-3, activation_function='sigmoid'):
        self.x_train = x_train
        self.y_train = y_train
        self.batchSize = x_train.shape[0]

        self.activation_function = activation_function
        self.maxiter = maxiter
        self.lr = lr

        self.init_training()

        optimizer = Adam(self)
        self.p, self.costHistory = optimizer.train()

    def trainParareal(self, x_train, y_train, maxiter=int(1e3), lr=1e-3, activation_function='sigmoid', Ncoarse=4, Nparareal=-1):
        self.x_train = x_train
        self.y_train = y_train
        self.batchSize = x_train.shape[-1]  # for now only full gradient descent

        self.activation_function = activation_function
        self.maxiter = maxiter
        self.lr = lr

        self.init_training()  # initialize a, z, delta now that we know the batch size !!!

        optimizer = Parareal(self, Ncoarse, Nparareal)
        self.p, self.costHistory = optimizer.train()

    def trainParaFlowS(self, x_train, y_train, Ng, Nf, dT, dt=-1, maxiter=int(1e3), lr=1e-3, activation_function='sigmoid'):
        self.x_train = x_train
        self.y_train = y_train
        self.batchSize = x_train.shape[-1]

        self.activation_function = activation_function
        self.maxiter = maxiter
        self.lr = lr

        self.init_training()

        optimizer = ParaFlowS(self, Ng, dT, Nf, dt)
        self.p, self.costHistory = optimizer.train()

    def trainParaFlowS2(self, x_train, y_train, Ng, Nf, dT, dt=-1, maxiter=int(1e3), lr=1e-3, activation_function='sigmoid'):
        self.x_train = x_train
        self.y_train = y_train
        self.batchSize = x_train.shape[-1]

        self.activation_function = activation_function
        self.maxiter = maxiter
        self.lr = lr

        self.init_training()

        optimizer = ParaFlowS2(self, Ng, dT, Nf, dt)
        self.p, self.costHistory = optimizer.train()

    def trainParAdamFlowS(self, x_train, y_train, Ng, Nf, dT, dt=-1, maxiter=int(1e3), lr=1e-3, activation_function='sigmoid'):
        self.x_train = x_train
        self.y_train = y_train
        self.batchSize = x_train.shape[-1]

        self.activation_function = activation_function
        self.maxiter = maxiter
        self.lr = lr

        self.init_training()

        optimizer = ParAdamFlowS(self, Ng, dT, Nf, dt)
        self.p, self.costHistory = optimizer.train()

    def trainLinear(self, x_train, y_train, activation_function='sigmoid'):
        # self.x_train = x_train
        # self.y_train = y_train
        # self.batchSize = x_train.shape[-1]
        #
        # self.activation_function = activation_function
        #
        # self.init_training()

        optimizer = Linear(self)
        # print initial cost
        print('initial cost:', self.cost())
        # self.p = optimizer.train()
        optimizer.train()
        self.costHistory = self.cost()

    # _____ functions for network differentiation _____

    def init_training(self):
        self.z = [np.ones((self.shape[i + 1], self.batchSize)) for i in range(self.L - 1)]
        self.delta = [np.ones((self.shape[i + 1], self.batchSize)) for i in range(self.L - 1)]
        self.a = [np.ones((self.shape[i], self.batchSize)) for i in range(self.L)]

    # handle functions to reshape matrices and biases
    def W(self, i):
        # assert that i is in the range [0, L-1]
        assert i in range(self.L - 1)
        start = 0
        for j in range(i):
            start += np.prod(self.shapes[j])  # add buffer matrix
            start += self.shapes[j][0]  # add buffer bias
        end = start + np.prod(self.shapes[i])
        return self.p[start:end].reshape(self.shapes[i], order='F')

    def b(self, i):
        # assert that i is in the range [0, L-1]
        assert i in range(self.L - 1)
        start = 0
        for j in range(i):
            start += np.prod(self.shapes[j])  # add buffer matrix
            start += self.shapes[j][0]  # add buffer bias
        start += np.prod(self.shapes[i])  # add buffer matrix
        end = start + self.shapes[i][0]
        return self.p[start:end].reshape(self.shapes[i][0])  # bias is a vector

    def dLp(self, i, Wnew, bnew):
        # assert that i is in the range [0, L-1]
        assert i in range(self.L - 1)
        # assert that Wnew and bnew have the right shape
        assert np.array(Wnew.shape).all() == self.shapes[i].all()
        assert bnew.shape[0] == self.shapes[i][0]

        start = 0
        for j in range(i):
            start += np.prod(self.shapes[j])  # add buffer matrix
            start += self.shapes[j][0]  # add buffer bias
        end = start + np.prod(self.shapes[i])
        self.dLp_[start:end] = Wnew.reshape(-1, order='F')
        start = end
        end = start + self.shapes[i][0]
        self.dLp_[start:end] = bnew

    def forwardPass(self):
        self.a[0] = self.x_train.reshape(-1, self.batchSize)
        for l in range(self.L - 1):
            self.z[l] = self.W(l) @ self.a[l] + self.b(l)[:, np.newaxis]
            self.a[l + 1] = self.sigma(self.z[l])

    def backwardPass(self, arg='L'):
        if arg == 'L':
            self.delta[self.L - 2] = (self.a[self.L - 1] - self.y_train) * self.sigmaPrime(self.z[self.L - 2])
        elif arg == 'F':
            self.delta[self.L - 2] = self.sigmaPrime(self.z[self.L - 2])
        else:
            raise ValueError('arg must be either L or F')

        for l in range(self.L - 2, 0, -1):
            self.delta[l - 1] = np.dot(self.W(l).T, self.delta[l]) * self.sigmaPrime(self.z[l - 1])

    def grad(self):
        for l in range(self.L - 1):
            self.dLp(l, np.dot(self.delta[l], self.a[l].T) / self.batchSize, np.mean(self.delta[l], axis=1))

    def diffL(self):
        self.forwardPass()
        self.backwardPass('L')
        self.grad()

    def diffF(self):
        self.forwardPass()
        self.backwardPass('F')
        self.grad()

    def cost(self):
        # return the cost using the Frobenius norm
        return 0.5 * np.linalg.norm(self.a[self.L - 1] - self.y_train, ord='fro') ** 2 / self.batchSize

    # _____ functions for prediction _____

    def predictRegression(self, x_test):
        # n = x_test.shape[-1]  # number of test points
        n = x_test.shape[0]
        y_test = np.ones(n)
        for i in range(n):

            if len(x_test.shape) == 1:
                x_i = np.array([[x_test[i]]]) # for 1D regression
            else:
                x_i = np.array([x_test[i]]).T #for multidimensional regression

            for l in range(self.L - 1):
                x_i = self.sigma(self.W(l) @ x_i + self.b(l)[:, np.newaxis])
            y_test[i] = x_i

        return y_test

    #todo: write functions to read and save weights