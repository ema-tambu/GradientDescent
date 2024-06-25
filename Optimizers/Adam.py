from seed import Seed
import numpy as np

np.random.seed(Seed().getSeed())


class Adam:
    def __init__(self, NN):
        self.NN = NN

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = 0 # self.NN.dLp_.copy()
        self.v = 0 # self.NN.dLp_.copy()
        self.t = 0

    def AdamStep(self):
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.NN.dLp_
        self.v = self.beta2 * self.v + (1 - self.beta2) * (self.NN.dLp_ ** 2)
        mHat = self.m / (1 - (self.beta1 ** self.t))
        vHat = self.v / (1 - (self.beta2 ** self.t))
        self.NN.p -= self.NN.lr * (mHat / (np.sqrt(vHat) + self.epsilon))
        # print('direction = ', (mHat / (np.sqrt(vHat) + self.epsilon)) )

    def train(self):
        costHistory = np.ones(int(self.NN.maxiter))
        temp = 0.1

        best_global_index = 0
        best_weights = np.copy(self.NN.p)

        self.t = 0
        converged = False
        for i in range(int(self.NN.maxiter)):
            self.t += 1
            self.NN.diffL()
            # print('dlp', self.NN.dLp_)
            self.AdamStep()
            costHistory[i] = self.NN.cost()
            if i == int(self.NN.maxiter * temp):
                print(int(temp * 100), '%')
                temp += 0.1

            if costHistory[i] < costHistory[best_global_index]:
                best_global_index = i
                best_weights = np.copy(self.NN.p)

            # Stopping criterion
            if costHistory[i] < 1e-4:
                print("Adam converged at iteration ", i, ", with cost ", costHistory[i])
                converged = True
                break

        if not converged:
            print("Adam did not converge after ", self.NN.maxiter, " iterations")
        print("Best global index: ", best_global_index)

        # return self.NN.p, costHistory
        return best_weights, costHistory
