def basis_Leihui(local_pos, edge_index, k=None, orientation=None):
    """Estimates a tangent basis for each point, given a k-nn graph and positions.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        k (int, optional): the number of neighbors per point,
            is derived from edge_index when no k is provided (default: None).
        orientation (Tensor, optional): an [N, 3] tensor with a rough direction of the normal to
            orient the estimated normals.
    """
    #row, col = edge_index
    #k = (row == 0).sum() if k is None else k
    #row, col = row.reshape(-1, k), col.reshape(-1, k)
    local_pos = local_pos.transpose(0, 2, 1)
    
    # SVD to estimate bases
    U, _, _ = LA.svd(local_pos)
    
    # Normal corresponds to smallest singular vector and normalize
    normal = U[:, :, 2]
    normal = normal / LA.norm(normal, axis=-1, keepdims=True).clip(EPS)

    # If normals are given, orient using the given normals
    if orientation is not None:
        normal = np.where(batch_dot(normal, orientation) < 0, -normal, normal)

    # X axis to largest singular vector and normalize
    x_basis = U[:, :, 0]
    x_basis = x_basis / LA.norm(x_basis, axis=-1, keepdims=True).clip(EPS)
    
    # Create orthonormal basis by taking cross product
    y_basis = np.cross(normal, x_basis)
    y_basis = y_basis / LA.norm(y_basis, axis=-1, keepdims=True).clip(EPS)
    
    return normal, x_basis, y_basis