from seed import Seed
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(Seed().getSeed())


class ParaFlowS2:
    def __init__(self, NN, Ng, dT, Nf, dt=-1):
        self.NN = NN
        self.Ng = Ng
        self.dT = dT
        self.Nf = Nf
        # for the small step size we can use the learning rate
        self.dt = dt
        if dt == -1:
            self.dt = self.NN.lr

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

    def train(self):
        costHistory = np.ones(self.NN.maxiter)
        costHistoryCoarse = np.ones(self.Ng - 1)

        printFlag = 0.1

        restart = 4000

        # zeroth iteration
        dim = len(self.NN.p)
        Ucoarse = np.zeros((self.Ng, dim))  # np.zeros((self.Ng + 1, dim))
        Uold = np.zeros((self.Ng, dim))  # np.zeros((self.Ng + 1, dim))
        Ufine = np.zeros(dim)
        # Ucoarse[0, :] = np.copy(self.NN.p)
        Uold[0, :] = np.copy(self.NN.p)

        # zeroth iteration
        for i in range(self.Ng - 1):
            Uold[i + 1, :] = np.copy(self.G_EE(Uold[i, :]))

        Ucoarse = np.copy(Uold)

        # ParaFlowS loop
        best_index = self.Ng - 1
        best_global_index = 0
        best_weights = np.copy(self.NN.p)
        flag = False
        for k in range(int(self.NN.maxiter / self.Ng) + 1):

            # restart
            if k % restart == 0 and k != 0:
                print('Restarting at iteration ', k)
                Ucoarse = np.zeros((self.Ng, dim))
                Uold = np.zeros((self.Ng, dim))
                # Ucoarse[0, :] = np.copy(best_weights)
                Uold[0, :] = np.copy(best_weights)
                for i in range(self.Ng - 1):
                    Uold[i + 1, :] = np.copy(self.G_EE(Uold[i, :]))
                    costHistoryCoarse[i] = self.NN.cost()
                Ucoarse = np.copy(Uold)
                # best_index = np.argmin(costHistoryCoarse)
                # best_weights = np.copy(Ucoarse[best_index, :])

            # fine iteration (no loop)
            Ufine = np.copy(self.F_EE(Ucoarse[best_index, :]))

            # prediction - correction
            for i in range(self.Ng - 1):
                temp = np.copy(self.G_EE(Ucoarse[i, :]))
                Ucoarse[i + 1, :] = np.copy(temp + Ufine - Uold[i + 1, :])
                Uold[i + 1, :] = np.copy(temp)
                costHistoryCoarse[i] = self.NN.cost()

            # find the minimum of the previous iteration
            best_index = np.argmin(costHistoryCoarse)
            costHistory[k * (self.Ng - 1): (k + 1) * (self.Ng - 1)] = costHistoryCoarse

            # keep the best global weights
            if costHistoryCoarse[best_index] < costHistory[best_global_index]:
                best_global_index = k * (self.Ng - 1) + best_index
                best_weights = np.copy(Ucoarse[best_index, :])
                # update initial weights, moving window (maybe we can drop best_weights)
                # Ucoarse[0, :] = np.copy(best_weights)
                # Ucoarse[0, :] = np.copy(Ucoarse[best_index, :])

            # if k == int(niter * printFlag):
            #     print(int(printFlag * 100), '%')
            #     print('best global index:', best_global_index)
            #     print('best global cost:', costHistory[best_global_index])
            #     printFlag += 0.1

            # Stopping criterion
            if costHistory[best_global_index] < 1e-3:
                print("k = ", k)
                print("ParaFlowS2 converged at iteration ", (self.Nf + self.Ng) * k, ", with cost ",
                      costHistory[best_global_index])
                flag = True
                break

        if not flag:
            print("ParaFlowS2 did not converge after ", self.NN.maxiter, " iterations.")

        # self.NN.p = np.copy(Ucoarse[best_index, :])
        print('best global index:', best_global_index)
        print('best global cost:', costHistory[best_global_index])
        self.NN.p = np.copy(best_weights)
        return self.NN.p, costHistory
