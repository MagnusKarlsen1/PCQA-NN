def coords_projected(local_pos, normal, x_basis, y_basis, k=None):
    """Projects neighboring points to the tangent basis
    and returns the local coordinates.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        normal (Tensor): an [N, 3] tensor with normals per point.
        x_basis (Tensor): an [N, 3] tensor with x basis per point.
        y_basis (Tensor): an [N, 3] tensor with y basis per point.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        k (int): the number of neighbors per point.
    """
    k = np.size(local_pos, 0) if k is None else k

    # Compute coords
    normal = np.tile(normal[:, None], (1, k, 1)).reshape(-1, 3)
    x_basis = np.tile(x_basis[:, None], (1, k, 1)).reshape(-1, 3)
    y_basis = np.tile(y_basis[:, None], (1, k, 1)).reshape(-1, 3)
    local_pos = local_pos - normal * batch_dot(local_pos, normal)
    x_pos = batch_dot(local_pos, x_basis).flatten()
    y_pos = batch_dot(local_pos, y_basis).flatten()
    coords = np.stack([x_pos, y_pos], axis=1)

    return coords