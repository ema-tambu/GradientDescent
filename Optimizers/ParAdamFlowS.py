from seed import Seed
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(Seed().getSeed())


class ParAdamFlowS:
    def __init__(self, NN, Ng, dT, Nf, dt=-1):
        self.NN = NN
        self.Ng = Ng
        self.dT = dT
        self.Nf = Nf
        # for the small step size we can use the learning rate
        self.dt = dt
        if dt == -1:
            self.dt = self.NN.lr

        # initialize parameters for Adam fine operator
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = 0
        self.v = 0
        self.t = 0

        # adam percentage usefulness
        # self.adam_percentage = []

    def G_EE(self, y):
        # assert p and y have the same shape
        assert np.array(y.shape).all() == np.array(self.NN.p.shape).all()
        self.NN.p = np.copy(y)
        self.NN.diffL()
        return self.NN.p - self.dT * self.NN.dLp_

    def F_EE(self, y):
        assert np.array(y.shape).all() == np.array(self.NN.p.shape).all()
        self.NN.p = np.copy(y)
        for i in range(self.Nf):
            self.NN.diffL()
            self.NN.p -= self.dt * self.NN.dLp_
        return self.NN.p

    # def G_Adam(self, y, dT):
    #     # assert p and y have the same shape
    #     assert np.array(y.shape).all() == np.array(self.NN.p.shape).all()
    #     self.NN.p = np.copy(y)
    #     self.NN.diffL()

    def G_Adam(self, y, t):
        t += 1
        assert np.array(y.shape).all() == np.array(self.NN.p.shape).all()
        self.NN.p = np.copy(y)

        costTemp = np.ones(self.Nf)
        best_index = 0
        best_weights = np.copy(self.NN.p)
        # self.m = 1
        # self.v = 1
        self.NN.diffL()
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.NN.dLp_
        self.v = self.beta2 * self.v + (1 - self.beta2) * (self.NN.dLp_ ** 2)
        mHat = self.m / (1 - (self.beta1 ** t))
        vHat = self.v / (1 - (self.beta2 ** t))
        self.NN.p -= self.dt * (mHat / (np.sqrt(vHat) + self.epsilon))
        return self.NN.p

    def F_Adam(self, y):
        assert np.array(y.shape).all() == np.array(self.NN.p.shape).all()
        self.NN.p = np.copy(y)

        costTemp = np.ones(self.Nf)
        best_index = 0
        best_weights = np.copy(self.NN.p)
        self.m = 0
        self.v = 0
        self.t = 0
        for i in range(self.Nf):
            self.t += 1
            self.NN.diffL()
            self.m = self.beta1 * self.m + (1 - self.beta1) * self.NN.dLp_
            self.v = self.beta2 * self.v + (1 - self.beta2) * (self.NN.dLp_ ** 2)
            mHat = self.m / (1 - (self.beta1 ** self.t))
            vHat = self.v / (1 - (self.beta2 ** self.t))
            self.NN.p -= self.dt * (mHat / (np.sqrt(vHat) + self.epsilon))
            costTemp[i] = self.NN.cost()
            if costTemp[i] < costTemp[best_index]:
                best_index = i
                best_weights = np.copy(self.NN.p)

        # self.adam_percentage.append(best_index / self.Nf)
        return best_weights

    # ParaFlowS v1.0
    # def train(self):
    def train_version_1_0(self):
        costHistory = np.ones(self.NN.maxiter)
        costHistoryCoarse = np.ones(self.Ng - 1)
        dim = len(self.NN.p)
        Ucoarse = np.zeros((self.Ng, dim))

        Ucoarse[0, :] = np.copy(self.NN.p)
        best_indexes = []

        # ParaFlowS loop
        flag = False
        for k in range(self.NN.maxiter):
            gk = np.copy(self.G_EE(Ucoarse[0, :]))
            # self.m = 0
            # self.v = 0
            # gk = np.copy(self.G_Adam(Ucoarse[0, :], k))

            # fk = np.copy(self.F_EE(Ucoarse[0, :]))
            fk = np.copy(self.F_Adam(Ucoarse[0, :]))

            # prediction - correction
            self.m = 0
            self.v = 0
            for i in range(self.Ng - 1):
                temp = np.copy(self.G_EE(Ucoarse[i, :]))
                # temp = np.copy(self.G_Adam(Ucoarse[i, :], k))
                Ucoarse[i + 1, :] = np.copy(temp + fk - gk)
                costHistoryCoarse[i] = self.NN.cost()

            # find the minimum achieved with the current iteration (excluding the first element)
            best_index = np.argmin(costHistoryCoarse[1:]) + 1
            best_indexes.append(best_index)
            costHistory[k] = costHistoryCoarse[best_index]

            Ucoarse[0, :] = np.copy(Ucoarse[best_index, :])  # reinsert in the first element the best solution

            # if k == int(niter * printFlag):
            #     print(int(printFlag * 100), '%')
            #     print('best global index:', best_global_index)
            #     print('best global cost:', costHistory[best_global_index])
            #     printFlag += 0.1

            # Stopping criterion
            if costHistory[k] < 1e-4:
                print("k = ", k)
                print("ParAdamFlowS converged at iteration ", k * (self.Nf + self.Ng + 1), ", with cost ",
                      costHistory[k])
                flag = True
                break

        if not flag:
            print("ParAdamFlowS did not converge after ", self.NN.maxiter * (self.Nf + self.Ng + 1), " iterations.")

        print('average best index: ', np.mean(best_indexes))

        self.NN.p = np.copy(Ucoarse[0, :])
        return self.NN.p, costHistory

    # ParaFlowS v 2.0
    def train(self):
    # def train_version_2_0(self):
        costHistory = np.ones(self.NN.maxiter)
        costHistoryCoarse = np.ones(self.Ng - 1)

        # zeroth iteration
        dim = len(self.NN.p)
        Ucoarse = np.zeros((self.Ng, dim))
        Uold = np.zeros((self.Ng, dim))
        Ufine = np.zeros(dim)
        # Ucoarse[0, :] = np.copy(self.NN.p)
        Uold[0, :] = np.copy(self.NN.p)

        # self.m = 0
        # self.v = 0
        # zeroth iteration
        for i in range(self.Ng - 1):
            Uold[i + 1, :] = np.copy(self.G_EE(Uold[i, :]))
            # Uold[i + 1, :] = np.copy(self.G_Adam(Uold[i, :], i))

        Ucoarse = np.copy(Uold)

        # ParaFlowS loop
        best_index = self.Ng - 1
        best_global_index = 0
        best_weights = np.copy(self.NN.p)
        flag = False
        for k in range(self.NN.maxiter):
            # Ufine = np.copy(self.F_EE(Ucoarse[best_index, :]))
            Ufine = np.copy(self.F_Adam(Ucoarse[best_index, :]))

            # prediction - correction
            # self.m = 0
            # self.v = 0
            for i in range(self.Ng - 1):
                temp = np.copy(self.G_EE(Ucoarse[i, :]))
                # temp = np.copy(self.G_Adam(Ucoarse[i, :], i))
                Ucoarse[i + 1, :] = np.copy(temp + Ufine - Uold[i + 1, :])
                Uold[i + 1, :] = np.copy(temp)
                costHistoryCoarse[i] = self.NN.cost()

            # find the minimum of the previous iteration
            best_index = np.argmin(costHistoryCoarse)

            # costHistory[k * (self.Ng - 1): (k + 1) * (self.Ng - 1)] = costHistoryCoarse
            costHistory[k] = costHistoryCoarse[best_index]

            # keep track of the best global weights
            if costHistoryCoarse[best_index] < costHistory[best_global_index]:
                best_global_index = k
                best_weights = np.copy(Ucoarse[best_index, :])

            # Stopping criterion
            if costHistory[best_global_index] < 1e-4:
                print("k = ", k)
                print("ParaFlowS converged at iteration ", (self.Nf + self.Ng) * k + self.Ng, ", with cost ",
                      costHistory[best_global_index])
                flag = True
                break

        if not flag:
            print("ParaFlowS did not converge after ", (self.Nf + self.Ng) * self.NN.maxiter + self.Ng, " iterations.")

        print('best global index:', best_global_index)
        print('best global cost:', costHistory[best_global_index])
        self.NN.p = np.copy(best_weights)
        return self.NN.p, costHistory
