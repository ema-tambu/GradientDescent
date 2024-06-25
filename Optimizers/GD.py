from seed import Seed
import numpy as np

np.random.seed(Seed().getSeed())


class GradientDescent:
    def __init__(self, NN):
        self.NN = NN
        # self.maxiter = maxiter
        # self.shape = shape
        # self.L = len(self.shape)
        # self.lr = lr
        # self.batchSize = batchSize
        # self.x_train = x_train
        # self.y_train = y_train
        # self.sigma = sigma
        # self.sigmaPrime = sigmaPrime

    def gradientStep(self):
        self.NN.p -= self.NN.lr * self.NN.dLp_

    def train(self):
        costHistory = np.empty(int(self.NN.maxiter))
        temp = 0.1

        for i in range(int(self.NN.maxiter)):
            self.NN.diffL()
            self.gradientStep()
            costHistory[i] = self.NN.cost()
            if i == int(self.NN.maxiter * temp):
                print(int(temp * 100), '%')
                temp += 0.1

            # Stopping criterion
            if costHistory[i] < 1e-4:
                print("Gradient Descent converged at iteration ", i, ", with cost ", costHistory[i])
                break

        return self.NN.p, costHistory
