from seed import Seed
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(Seed().getSeed())


class Parareal:
    def __init__(self, NN, Ncoarse, Nparareal=-1):
        self.NN = NN
        self.Ncoarse = Ncoarse
        self.T = self.NN.maxiter * self.NN.lr
        self.Nfine = int(self.NN.maxiter / self.Ncoarse)
        assert self.Nfine > 0
        self.Nparareal = self.Ncoarse
        if Nparareal != -1:
            self.Nparareal = Nparareal

    def G(self, y, dT):
        # assert p and y have the same shape
        assert np.array(y.shape).all() == np.array(self.NN.p.shape).all()
        self.NN.p = np.copy(y)
        self.NN.diffL()
        return self.NN.p - dT * self.NN.dLp_

    def F(self, y):
        self.NN.p = np.copy(y)
        for i in range(self.Nfine):
            self.NN.diffL()
            self.NN.p -= self.NN.lr * self.NN.dLp_
        return self.NN.p

    def plot(self, vector, title):
        plt.semilogy(vector, linestyle='dashed', marker='o')
        i = 0
        while i < self.Ncoarse ** 2:
            i += self.Ncoarse
            plt.axvline(i, color='gray', linestyle='dotted')
        plt.title(title)
        plt.show()

    def train(self):
        costHistory = np.ones(int(self.Ncoarse ** 2))
        costHistoryCoarse = np.ones(self.Ncoarse)

        dT = self.T / self.Ncoarse
        dim = len(self.NN.p)
        Ucoarse = np.zeros((self.Ncoarse + 1, dim))
        Uold = np.zeros((self.Ncoarse + 1, dim))
        Ufine = np.zeros((self.Ncoarse, dim))
        Ucoarse[0, :] = np.copy(self.NN.p)
        Uold[0, :] = np.copy(self.NN.p)

        # zeroth iteration
        for i in range(self.Ncoarse):
            Uold[i + 1, :] = np.copy(self.G(Uold[i, :], dT))

        Ucoarse = np.copy(Uold)

        bestindex = self.Ncoarse + 1

        # parareal loop
        for k in range(self.Nparareal):

            print('Parareal iteration: ', k + 1, ' / ', self.Ncoarse)

            # parallel step
            for i in range(self.Ncoarse):
                Ufine[i, :] = np.copy(self.F(Ucoarse[i, :]))

            # prediction correction
            for i in range(self.Ncoarse):
                temp = np.copy(self.G(Ucoarse[i, :], dT))
                Ucoarse[i + 1, :] = np.copy(temp + Ufine[i, :] - Uold[i + 1, :])
                Uold[i + 1, :] = np.copy(temp)
                costHistoryCoarse[i] = self.NN.cost()

            if k == self.Nparareal - 1:
                # find the set of parameters that minimizes the loss function the most
                bestindex = np.argmin(costHistoryCoarse)

            costHistory[k * self.Ncoarse: (k + 1) * self.Ncoarse] = costHistoryCoarse
            self.plot(costHistory, 'Cost history')

        self.NN.p = np.copy(Ucoarse[bestindex, :])

        return self.NN.p, costHistory
