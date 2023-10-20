import sys

import jax
import jax.numpy as jnp

from tessellate_ipu.linalg.tile_linalg_hessenberg import ipu_hessenberg
from tessellate_ipu.linalg.tile_linalg_tridiagonal_eigenvalue import ipu_tridiagonal_eigenvalue
from tessellate_ipu.linalg.tile_linalg_tridiagonal_solver import ipu_tridiag_solve

jax.config.FLAGS.jax_platform_name = "cpu"
jax.config.update("jax_enable_x64", False)


def ipu_tridiagonal_eigh(d, e, n_iter=2, seed=42):

    N = d.shape[0]
    eig = ipu_tridiagonal_eigenvalue(d, e)[:N]

    diag = jnp.tile(d.reshape(1, -1), (N, 1)) - eig[:, jnp.newaxis]

    udiag = jnp.concatenate([e, jnp.array([0], dtype=jnp.float32)]).reshape(1, -1)
    udiag = jnp.tile(udiag, (N, 1))

    ldiag = jnp.concatenate([jnp.array([0], dtype=jnp.float32), e]).reshape(1, -1)
    ldiag = jnp.tile(ldiag, (N, 1))

    prng_key = jax.random.PRNGKey(seed)
    x = jax.random.normal(prng_key, shape=(N, diag.shape[1]), dtype=jnp.float32)
    x /= jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]

    def inverse_iteration(i, x):
        x = ipu_tridiag_solve(diag, udiag, ldiag, x)
        x /= jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]
        return x

    x = jax.lax.fori_loop(0, n_iter, inverse_iteration, x)

    return x, eig


def ipu_eigh_hess(M):

    Q, M_tri_ = ipu_hessenberg(M)
    M_tri = M_tri_.array
    d, e = jnp.diag(M_tri), jnp.diag(M_tri, k=1)

    x, eig = ipu_tridiagonal_eigh(d, e)
    return x @ Q.array.T, eig


if __name__ == "__main__":
    import argparse

    import jax
    import numpy as np
    import scipy
    from icecream import ic

    parser = argparse.ArgumentParser(description="Test for IPU eigh_tridiagonal")
    parser.add_argument(
        "-r",
        "--random",
        help="run test on a random matrix of size x-by-x computing only y eigenvectors/eigenvalues",
        type=str,
        nargs=2,
    )
    parser.add_argument(
        "-f", "--file", help="run test on a matrix constructed from L_inv and H npz files at the given path", type=str
    )

    cfg = parser.parse_args()

    if cfg.file and cfg.random:
        print("Specify only one of the '-r' and '-f' options")
        sys.exit(1)

    seed = 42
    np.random.seed(seed)

    np.set_printoptions(precision=3, linewidth=120, suppress=True)

    if cfg.random:
        S = int(cfg.random[0])
        N = int(cfg.random[1])
        mat = np.random.rand(S, S).astype(np.float32)
        mat = (mat + mat.T) / 2
    elif cfg.file:
        L_inv = np.load(cfg.file + "L_inv.npz")["v"]
        H = np.load(cfg.file + "H.npz")["v"]
        mat = L_inv @ H @ L_inv.T

        S = mat.shape[0]
        N = S
        ic(S)
    else:
        print("Specify one of the options -r or -f")
        sys.exit(1)

    x, eig = jax.jit(ipu_eigh_hess, backend="ipu")(mat)

    x = np.array(x)

    ic("IPU:")
    ic(np.max(np.abs(mat @ x.T - eig * x.T)))
    ic(np.max(np.abs(x.T @ np.diag(eig) @ x - mat)))

    ic("Eigenvector orthogonality:")
    ic(np.max(np.abs(x.T @ x - np.eye(x.shape[0]))))

    D, Q = np.linalg.eigh(mat)
    ic("Numpy linalg.eigh")
    ic(np.max(np.abs(Q @ np.diag(D) @ Q.T - mat)))
    ic(np.max(np.abs(mat @ Q - D * Q)))

    # compute eigh using hessenberg then eigh_tridiagonal.

    T, Q = scipy.linalg.hessenberg(mat, calc_q=True)

    d, e = np.diag(T), np.diag(T, k=1)

    w, vtd = scipy.linalg.eigh_tridiagonal(d, e)
    v = Q @ vtd
    ic("Scipy hessenberg + linalg.eigh_tridiagonal")
    ic(np.max(np.abs(mat @ v - v @ np.diag(w))))
    ic(np.max(np.abs(T @ vtd - vtd @ np.diag(w))))
