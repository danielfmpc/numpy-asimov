import numpy as np

a = np.array([[3,5,5], [2,0,1]])
b = np.array([4,5,-2])

np.matmul(a,b)

a.ndim # number of dimensions (2)
a.shape # shape of array (2,3)
a.dtype # data type of array (int64)
a.itemsize # number of elements in array of bytes (8)
a.size # number of elements in array (6)

idades = np.array([10, 12, 14, 16, 18], dtype=np.int8)
idades.dtype

#-----------------------------------------------------------------------------------------------------------------------

