# Linear Algebra algorithms benchmarking

GPU benchmarks ran on Google Colab A100(40GB).

# QR algorithm

| Size  | PyTorch (A100) | JAX (A100) | JAX IPU Mk2 Bow | Poplibs IPU Mk2 Bow |
| ----- | ------------:  | --------:  | -------------:  | ------------------: |
| 32    | 0.14ms         | 0.19ms     | 0.027ms (49k)   | 0.09ms (162k)
| 64    | 0.30ms         | 0.33ms     | 0.10ms (189k)   | 0.14ms (245k)
| 128   | 0.75ms         | 0.79ms     | 0.21ms (372k)   | 0.24ms (424k)
| 256   | 1.20ms         | 1.28ms     | 0.34ms (610k)   | 0.46ms (825k)
| 512   | 2.88ms         | 2.98ms     | 0.94ms (1.69M)  | 1.03ms (1.86M)
| 1024  | 6.92ms         | 7.16ms     | -               | 3.6ms (6.5M)
| 2048  | 17.7ms         | 18.0ms     | -               | 15ms (26.9M)

Initial input matrix sharding across tiles: additional ~10%.

# Eigh algorithm

Testing 6 iterations of Jacobi algorithm on IPU.

| Size  | PyTorch (A100) | JAX (A100) | JAX IPU Mk2 Bow
| ----- | ------------:  | ---------: | -------------:
| 32    | 0.53ms         | 0.78ms     | 0.19ms (335k)
| 64    | 1.01ms         | 2.18ms     | 0.53ms (960k)
| 128   | 2.45ms         | 4.49ms     | 1.55ms (2.8M)
| 256   | 6.58ms         | 8.46ms     | 5.33ms (9.6M)
| 512   | 18.3ms         | 25.4ms     | -
| 1024  | 18.4ms         | 67.1ms     | -
| 2048  | 41.6ms         | 147ms      | -
