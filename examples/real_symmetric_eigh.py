import sys

import jax
import jax.numpy as jnp
import numpy as np
from icecream import ic

from tessellate_ipu.linalg import ipu_hessenberg, ipu_tridiag_solve
from tessellate_ipu.linalg.tile_linalg_tridiagonal_eigh import ipu_eigh_tridiagonal

jax.config.FLAGS.jax_platform_name = "cpu"
jax.config.update("jax_enable_x64", False)

if len(sys.argv) != 3:
    print(sys.argv[0] + " <size> <num eigenvectors>")
    sys.exit(1)

seed = 42
np.random.seed(seed)

np.set_printoptions(precision=3, linewidth=120, suppress=True)


S = int(sys.argv[1])
N = int(sys.argv[2])

mat = np.random.rand(S, S).astype(np.float32)
mat = (mat + mat.T) / 2


def real_symmetric_eigh(M):

    Q, M_tri_ = ipu_hessenberg(M)
    M_tri = M_tri_.array

    d, e = jnp.diag(M_tri), jnp.diag(M_tri, k=1)
    eig = ipu_eigh_tridiagonal(d, e)[:N]

    diag = jnp.tile(jnp.diag(M_tri, k=0).reshape(1, -1), (N, 1)) - eig[:, jnp.newaxis]

    udiag = jnp.concatenate([jnp.diag(M_tri, k=1), jnp.array([0], dtype=jnp.float32)]).reshape(1, -1)
    udiag = jnp.tile(udiag, (N, 1))

    ldiag = jnp.concatenate([jnp.array([0], dtype=jnp.float32), jnp.diag(M_tri, k=-1)]).reshape(1, -1)
    ldiag = jnp.tile(ldiag, (N, 1))

    prng_key = jax.random.PRNGKey(seed)
    x = jax.random.normal(prng_key, shape=(N, diag.shape[1]), dtype=jnp.float32)
    x /= jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]

    def inverse_iteration(i, x):
        x = ipu_tridiag_solve(diag, udiag, ldiag, x)
        x /= jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]
        return x

    x = jax.lax.fori_loop(0, 2, inverse_iteration, x)

    return x @ Q.array.T, eig


x, eig = jax.jit(real_symmetric_eigh, backend="ipu")(mat)

# ic(x)

# ic(mat @ eigv - eig * eigv)
# ic(mat @ x.T - eig * x.T)

ic(np.max(np.abs(mat @ x.T - eig * x.T)))
