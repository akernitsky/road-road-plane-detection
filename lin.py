import numpy as np


data = np.array([[20, 10, 10], [0, -10, 10], [0, 0, 0]])

A = np.c_[data[0], data[1], np.ones(data[0].shape)]
print(np.linalg.lstsq(A, data[2], rcond=None))   # coefficients

#print(C)

#1 * x + -2 * y + z + d * 0  = 0



#-0.01 * x + 0 * y + 0 * d + z = 0