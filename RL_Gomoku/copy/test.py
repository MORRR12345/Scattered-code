import torch

matrix = torch.rand(3, 4)
matrix_2 = matrix
matrix_2[0,0] = 0
print(matrix)
print(matrix_2)
