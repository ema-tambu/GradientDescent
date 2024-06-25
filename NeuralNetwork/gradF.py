from seed import Seed
import numpy as np

np.random.seed(Seed().getSeed())


class gradF:
    def __init__(self, NN):
        self.NN = NN
        # dimension of dFp_ should be (len(p) x batchSize)
        self.dFp_ = np.zeros((np.shape(self.NN.p)[0], self.NN.batchSize))
        # dimension of the regression task we're doing
        try:
            self.dim = self.NN.x_train.shape[1]
        except:
            self.dim = 1

    def dFp(self, i, Wnew, bnew, d):
        assert i in range(self.NN.L - 1)
        # assert that Wnew and bnew have the right shape
        assert np.array(Wnew.shape).all() == self.NN.shapes[i].all()
        assert bnew.shape[0] == self.NN.shapes[i][0]

        start = 0
        for j in range(i):
            start += np.prod(self.NN.shapes[j])  # add buffer matrix
            start += self.NN.shapes[j][0]  # add buffer bias
        end = start + np.prod(self.NN.shapes[i])
        self.dFp_[start:end, d] = Wnew.reshape(-1, order='F')
        start = end
        end = start + self.NN.shapes[i][0]
        self.dFp_[start:end] = bnew

    def forwardPass(self, d):
        assert d in range(self.NN.batchSize)
        # deal with multidimensional regression or not
        try:
            self.NN.a[0] = self.NN.x_train[:, d].reshape(-1, self.dim)
        except:
            self.NN.a[0] = self.NN.x_train[d].reshape(-1, 1)
        for l in range(self.NN.L - 1):
            # temp1 = self.NN.W(l)
            # temp2 = self.NN.a[l]
            # temp3 = temp1 @ temp2
            self.NN.z[l] = self.NN.W(l) @ self.NN.a[l] + self.NN.b(l)[:, np.newaxis]
            self.NN.a[l + 1] = self.NN.sigma(self.NN.z[l])

    def backwardPass(self):
        self.NN.delta[self.NN.L - 2] = self.NN.sigmaPrime(self.NN.z[self.NN.L - 2])
        for l in range(self.NN.L - 2, 0, -1):
            self.NN.delta[l - 1] = np.dot(self.NN.W(l).T, self.NN.delta[l]) * self.NN.sigmaPrime(self.NN.z[l - 1])

    def grad(self, d):
        assert d in range(self.NN.batchSize)
        for l in range(self.NN.L - 1):
            # self.NN.dLp(l, np.dot(self.NN.delta[l], self.NN.a[l].T) / self.NN.batchSize, np.mean(self.NN.delta[l], axis=1))
            Wnew = np.dot(self.NN.delta[l], self.NN.a[l].T)
            bnew = self.NN.delta[l]
            self.dFp(l, Wnew, bnew, d)

    def diff(self):
        self.NN.z = [np.ones(self.NN.shape[i + 1]) for i in range(self.NN.L - 1)]
        self.NN.delta = [np.ones(self.NN.shape[i + 1]) for i in range(self.NN.L - 1)]
        self.NN.a = [np.ones(self.NN.shape[i]) for i in range(self.NN.L)]
        for d in range(self.NN.batchSize):
            self.forwardPass(d)
            self.backwardPass()
            self.grad(d)
        print('debug')
        return self.dFp_
