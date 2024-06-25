from seed import Seed
import numpy as np
from NeuralNetwork.gradF import gradF

np.random.seed(Seed().getSeed())


class Linear:
    def __init__(self, NN):
        self.NN = NN
        self.T = 7    # time horizon (learning rate)

    def LinearStep(self):

        # compute $\nabla_\theta f(X,\theta_0)$
        differentiator = gradF(self.NN)
        grad_F = differentiator.diff() # (nt x ns)

        # compute $f(X,\theta_0) - Y$
        temp = self.NN.predictRegression(self.NN.x_train) - self.NN.y_train

        # compute vector q
        q = np.dot(grad_F, temp)

        print('dimension of gradF:', grad_F.shape)
        print('rank of gradF:', np.linalg.matrix_rank(grad_F))

        # compute matrix M
        M = np.dot(grad_F, grad_F.T) + (1e-3)*np.eye(grad_F.shape[0])
        print('rank of M:', np.linalg.matrix_rank(M))
        print('det of M:', np.linalg.det(M))

        # compute P = exp(MT)
        P = np.exp(M*self.T)

        # compute RHS $e^{Mt}(M\theta_0 - q) + \theta_0 - M\theta_0 + q$
        RHS = P @ (np.dot(M, self.NN.p) - q) + self.NN.p - np.dot(M, self.NN.p) + q

        # cpmpute Moore-Penrose pseudoinverse of P
        P_inv = np.linalg.pinv(P)

        print('rank of P:', np.linalg.matrix_rank(P))
        print('det of P:', np.linalg.det(P))
        print('max(P_inv):', np.max(P_inv))
        print('min(P_inv):', np.min(P_inv))

        # compute $\theta_1 = P^{-1}RHS$
        self.NN.p = np.copy(P_inv @ RHS)
        print('cost: ', self.NN.cost())

        return

    def NTKstep(self):
        # compute f(X,\theta) - Y
        temp = self.NN.predictRegression(self.NN.x_train) - self.NN.y_train

        # compute NTK: K_0(x_i, x_j)
        differentiator = gradF(self.NN)
        grad_F = differentiator.diff()
        NTK = np.dot(grad_F.T, grad_F)
        print('rank of NTK:', np.linalg.matrix_rank(NTK))
        print('det of NTK:', np.linalg.det(NTK))

        # try NTK with neural tangent library
        # 2D sphere not singular

        return
    def train(self):
        # return self.LinearStep()
        # for i in range(10):
        #     self.NN.p = self.LinearStep()
        # return self.NN.p
        return self.NTKstep()