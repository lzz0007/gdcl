import numba
import numpy as np
import scipy.sparse as sp
import time
from utils.configurator import Config
from utils.dataset import RecDataset


@numba.njit(cache=False, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            d = deg[vnode]
            if res_vnode >= alpha_eps * deg[vnode]: # look for neighbors with lower degree
                if vnode not in q:
                    q.append(vnode)

    return list(p.keys()), list(p.values())


@numba.njit(cache=False)
def calc_ppr(indptr, indices, deg, alpha, epsilon, nodes):
    js = []
    vals = []
    for i, node in enumerate(nodes):
        j, val = _calc_ppr_node(node, indptr, indices, deg, alpha, epsilon)
        js.append(j)
        vals.append(val)
    return js, vals


@numba.njit(cache=False, parallel=True)
def calc_ppr_topk_parallel(n_user, m_item, indptr, indices, deg, alpha_u, alpha_i, epsilon, nodes, topu, topi):
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        alpha = alpha_u if i < n_user else alpha_i
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon)
        j_np, val_np = np.array(j), np.array(val)

        idx_u = np.where(j_np < n_user)[0]
        idx_i = np.where(j_np >= n_user)[0]

        # topk based on u and i
        idx_topk_u = idx_u[np.argsort(val_np[idx_u])][-topu:]
        idx_topk_i = idx_i[np.argsort(val_np[idx_i])][-topi:]
        idx_topk = np.concatenate((idx_topk_u, idx_topk_i), axis=0)

        js[i] = j_np[idx_topk]
        vals[i] = val_np[idx_topk]
    return js, vals


def ppr_topk(n_user, m_item, adj_matrix, alpha_u, alpha_i, epsilon, nodes, topu, topi):
    """Calculate the PPR matrix approximately using Anderson."""

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    neighbors, weights = calc_ppr_topk_parallel(n_user, m_item, adj_matrix.indptr, adj_matrix.indices, out_degree,
                                                numba.float32(alpha_u), numba.float32(alpha_i), numba.float32(epsilon), nodes, topu, topi)

    return construct_sparse(neighbors, weights, (len(nodes), nnodes))


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


def topk_ppr_matrix(n_user, m_item, adj_matrix, alpha_u, alpha_i, eps, idx, topu, topi, normalization='row'):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix = ppr_topk(n_user, m_item, adj_matrix, alpha_u, alpha_i, eps, idx, topu, topi).tocsr()

    if normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1 # row sum of adj
        deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
        deg_inv_sqrt = 1. / deg_sqrt

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg_sqrt[idx[row]] * topk_matrix.data * deg_inv_sqrt[col]
    elif normalization == 'col':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_inv = 1. / np.maximum(deg, 1e-12)

        row, col = topk_matrix.nonzero()
        # assert np.all(deg[idx[row]] > 0)
        # assert np.all(deg[col] > 0)
        topk_matrix.data = deg[idx[row]] * topk_matrix.data * deg_inv[col]
    elif normalization == 'row':
        pass
    else:
        raise ValueError(f"Unknown PPR normalization: {normalization}")

    return topk_matrix

