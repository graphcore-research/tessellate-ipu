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

Q_ = np.array(Q.array)
R_ = np.array(R.array)
print("\nR matrix")
print(R_)
print("\nQ matrix")
print(Q_)
print(f"\nReconstruction Delta: {np.max(np.abs(Q_ @ R_ @ Q_.T - A))}")
print("\nQ.T @ Q")
print(Q_.T @ Q_)
