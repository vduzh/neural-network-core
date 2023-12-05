import random
import unittest

import numpy as np

from basic_inference import predict
from utils import sparse_cross_entropy, to_full


class TestTranning(unittest.TestCase):
    def setUp(self):
        self.INPUT_DIM = 4
        self.OUTPUT_DIM = 3
        self.H_DIM = 5

    def test_train(self):
        x = np.random.randn(1, self.INPUT_DIM)
        y = random.randint(0, self.OUTPUT_DIM)

        w1 = np.random.randn(self.INPUT_DIM, self.H_DIM)
        b1 = np.random.randn(1, self.H_DIM)
        w2 = np.random.randn(self.H_DIM, self.OUTPUT_DIM)
        b2 = np.random.randn(1, self.OUTPUT_DIM)

        # Forward
        z = predict(x, [w1, w2], [b1, b2])
        print('predict', z)
        e = sparse_cross_entropy(z, y)
        print('e', e)

        # Backward
        y_full = to_full(y, self.OUTPUT_DIM)
        dE_dt2 = z - y_full
        # dE_dW2 = h1.T @ dE_dt2
        # dE_db2 = dE_dt2
        # dE_dh1 = dE_dt2 @ W2.T

        # print('y_full', y_full)


if __name__ == '__main__':
    unittest.main()
