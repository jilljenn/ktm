from sklearn.utils.random import sample_without_replacement
import numpy as np


'''
for a in range(1, 5):
    for b in range(1, 5):
        print(a, b, 2 ** (a - 1) * (2 * b - 1))
'''

        
def sample_pairs(n, k):
    z = sample_without_replacement(n * (n - 1) / 2, k)
    # print(z)
    w = np.floor((-1 + np.sqrt(1 + 8 * z)) / 2).astype(int)
    # print(w)
    x = w * (w + 3) // 2 - z
    y = z - w * (w + 1) // 2
    return x, x + 1 + y


if __name__ == '__main__':
    print(sample_pairs(4, 3))
