import sys

import jax
import numpy as np

from tessellate_ipu.linalg import ipu_hessenberg

jax.config.FLAGS.jax_platform_name = "cpu"
jax.config.update("jax_enable_x64", False)


d = int(sys.argv[1])
np.random.seed(42)

np.set_printoptions(precision=3, linewidth=120, suppress=True)

A = np.random.normal(0, 1, (d, d))
A = (A + A.T) / 2

Q, R = jax.jit(ipu_hessenberg, backend="ipu")(A)

Q_ = Q.array.copy()
R_ = R.array.copy()
print("R matrix")
print(R_)
print("Q matrix (top left 6-by-6 corner)")
print(Q_)
print(f"\nDelta: {np.max(np.abs(Q_ @ R_ @ Q_.T - A))}")

print(Q_.T @ Q_)
