import numpy as np
from pcdiff import knn_graph, estimate_basis, build_grad_div, laplacian

data = np.loadtxt('C:/Users/aagaa/Documents/GitHub/R-D/Code/Leihui Code/src/data/bunny.xyz')
pos = data[:, 0:3]
# Random point cloud
#pos = np.random.rand(1000, 3)

Cloud_size = np.size(pos,0)

# Generate kNN graph
edge_index = knn_graph(pos, 10)
# Estimate normals and local frames
basis = estimate_basis(pos, edge_index)
# Build gradient and divergence operators (Scipy sparse matrices)
grad, div = build_grad_div(pos, *basis, edge_index)

# Setup the Laplacian as the divergence of gradient:
laplacian = -(div @ grad)

# Define some function on the point cloud
x = np.random.rand(Cloud_size, 1)

# Compute gradient of function
# The output is of size 2N, with the two components of the vector field interleaved:
# [x_1, y_1, x_2, y_2, ..., x_N, y_N]
grad_x = grad @ x

print('Gradients:\n', np.sum(grad.row==51),
      '\n\n Number of points: ', Cloud_size)