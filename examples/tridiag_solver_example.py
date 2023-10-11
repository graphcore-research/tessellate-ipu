import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import spdiags

from tessellate_ipu.linalg import ipu_tridiag_solve

jax.config.FLAGS.jax_platform_name = "cpu"
jax.config.update("jax_enable_x64", False)


N = int(sys.argv[1])
np.random.seed(42)

np.set_printoptions(precision=3, linewidth=120, suppress=True)


diag = np.arange(N).reshape(1, -1).astype(jnp.float32)
ldiag = np.random.rand(N - 1).reshape(1, -1).astype(jnp.float32)
rhs = np.random.rand(N).reshape(1, -1).astype(jnp.float32)

x_ = jax.jit(ipu_tridiag_solve, backend="ipu")(diag, ldiag, rhs)

x = np.array(x_.array)

T = spdiags([np.concatenate([np.array([0]), ldiag]), diag, np.concatenate([ldiag, [0]])], (1, 0, -1), N, N)

delta = T @ x - rhs
print(np.max(np.abs(delta)))
