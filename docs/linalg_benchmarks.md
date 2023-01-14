# Linear Algebra algorithms benchmarking

# QR algorithm

| Size  | PyTorch (A100) | JAX IPU Mk2 Bow
| ----- | ------------:  | -------------:
| 32    | 0.17ms         | -
| 64    | 0.36ms         |
| 128   | 0.84ms         |
| 256   | 1.20ms         |
| 512   | 2.88ms         |
| 1024  | 6.92ms         |
| 2048  | 17.7ms         |

# Eigh algorithm

| Size  | PyTorch (A100) | JAX IPU Mk2 Bow
| ----- | ------------:  | -------------:
| 32    | 0.61ms         | -
| 64    | 1.09ms         |
| 128   | 2.45ms         |
| 256   | 6.58ms         |
| 512   | 18.3ms         |
| 1024  | 18.4ms         |
| 2048  | 41.6ms         |
