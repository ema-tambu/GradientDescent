# from Tasks import ClassificationTask
import RegressionTask
import RegressionTask2D

from seed import Seed
import numpy as np


if __name__ == '__main__':
    i = 4
    while i < 15:
        j = 0.7
        while j < 1.4:
            k = 0.3
            while k < 1.0:
                print('i =', i, 'j = ', j, 'k = ', k)
                np.random.seed(Seed().getSeed())
                RegressionTask.regression_task1(
                    [1, 3, 3, 1], 'tanh',
                    # Ng=10, Nf=300, dT=1.2, lr=0.9, maxiter=60, plots=True)
                    optimizer='GD', lr=0.7, maxiter=int(5e5), plots=True)
                    # optimizer='ParAdamFlowS', Ng=4, Nf=255, dT=0.1, lr=0.01, maxiter=int(20), plots=True)
                    # optimizer='Adam', lr=0.01, maxiter=int(1e5), plots=True)
                # k += 0.1
                break
            # j += 0.1
            break
        # i += 1
        break
    # ClassificationTask.classification_task()
    # RegressionTask2D.regression_task2D([2, 16, 32, 64, 32, 16, 1], 'tanh')

# - set global tolerance for stopping criterion of all algorithm
# - test pest performance GD & adam and then from version 2.0 down