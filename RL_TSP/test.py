import numpy as np
import torch

matrix = np.random.rand(3, 4)
print(matrix)
end_pos = matrix[1:3, 0].reshape(2, 1)
print(end_pos)