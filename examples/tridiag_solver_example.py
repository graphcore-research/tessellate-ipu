import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import spdiags

from tessellate_ipu.linalg import ipu_tridiag_solve

jax.config.FLAGS.jax_platform_name = "cpu"
jax.config.update("jax_enable_x64", False)


N = int(sys.argv[2])
M = int(sys.argv[1])
np.random.seed(43)

np.set_printoptions(precision=3, linewidth=120, suppress=True)


diag = np.random.rand(M, N).astype(jnp.float32)
udiag = np.random.rand(M, N).astype(jnp.float32)
ldiag = np.roll(udiag, 1, axis=1)
rhs = np.random.rand(M, N).astype(jnp.float32)

x_ = jax.jit(ipu_tridiag_solve, backend="ipu")(diag, udiag, ldiag, rhs)

x = np.array(x_.array)

print(x.shape)

deltas = []
for i in range(M):
    data = np.vstack(
        # [np.roll(udiag[i].flat, 1, axis=0), diag[i].flat, udiag[i].flat],
        [np.roll(udiag[i].flat, 1, axis=0), diag[i].flat, np.roll(ldiag[i].flat, -1, axis=0)],
    )
    T = spdiags(data, (1, 0, -1), N, N).toarray()

    delta = T @ x[i].reshape(N, 1) - rhs[i].reshape(N, 1)

    deltas.append(delta)

print("Max abs delta:", np.max(np.abs(np.array(deltas))))
