# TessellateIPU supported operations

## [JAX LAX operations](https://jax.readthedocs.io/en/latest/jax.lax.html)

**Note:** Inplace operations are represented by additional primitives (e.g. `add_inplace_p`). For binary operations, only the first argument is supported for inplace update.

| Operation              | Supported          | Inplace            | Remarks |
| ---------------------- | ------------------ | ------------------ |-------- |
| `abs`                  | :white_check_mark: | :white_check_mark: |         |
| `add`                  | :white_check_mark: | :white_check_mark: |         |
| `acos`                 | :x:                | :x:                | Not directly available in Graphcore Poplibs        |
| `approx_max_k`         | :x:                | n/a                |         |
| `approx_min_k`         | :x:                | n/a                |         |
| `argmax`               | :x:                | n/a                |         |
| `argmin`               | :x:                | n/a                |         |
| `asin`                 | :white_check_mark: | :white_check_mark: |         |
| `atan`                 | :white_check_mark: | :white_check_mark: |         |
| `atan2`                | :white_check_mark: | :white_check_mark: |         |
| `batch_matmul`         | :x:                | n/a                | See `dot_general`         |
| `bessel_i0e`           | :x:                | n/a                |         |
| `bessel_i1e`           | :x:                | n/a                |         |
| `betainc`              | :x:                | n/a                |         |
| `bitcast_convert_type` | :white_check_mark: | :white_check_mark: | Only same size dtype supported. |
| `bitwise_not`          | :white_check_mark: | :x:                |         |
| `bitwise_and`          | :white_check_mark: | :x:                |         |
| `bitwise_or`           | :white_check_mark: | :x:                |         |
| `bitwise_xor`          | :white_check_mark: | :x:                |         |
| `population_count`     | :white_check_mark: | :x:                |         |
| `broadcast`            | :x:                | :x:                |         |
| `broadcast_in_dim`     | :x:                | :x:                |         |
| `cbrt`                 | :white_check_mark: | :white_check_mark: |         |
| `ceil`                 | :white_check_mark: | :white_check_mark: |         |
| `clamp`                | :white_check_mark: | :x:                |         |
| `collapse`             | :white_check_mark: | :white_check_mark: |         |
| `complex`              | :x:                | n/a                | Complex not supported in IPU XLA backend        |
| `concatenate`          | :white_check_mark: | n/a                |         |
| `conj`                 | :x:                | n/a                | Complex not supported in IPU XLA backend        |
| `conv`                 | :x:                | n/a                |         |
| `convert_element_type` | :white_check_mark: | :x:                |         |
| `conv_general_dilated` | :x:                | n/a                |         |
| `conv_transpose`       | :x:                | n/a                |         |
| `cos`                  | :white_check_mark: | :white_check_mark: |         |
| `cosh`                 | :white_check_mark: | :white_check_mark: |         |
| `cummax`               | :x:                | :x:                |         |
| `cummin`               | :x:                | :x:                |         |
| `cumprod`              | :x:                | :x:                |         |
| `cumsum`               | :x:                | :x:                |         |
| `digamma`              | :x:                | :x:                |         |
| `div`                  | :white_check_mark: | :white_check_mark: |         |
| `dot`                  | :white_check_mark: | n/a                |         |
| `dot_general`          | :white_check_mark: | n/a                | Limited set of configurations. See below. |
| `dynamic_slice`        | :x:                | n/a                |         |
| `dynamic_update_slice` | :x:                | n/a                |         |
| `eq`                   | :white_check_mark: | n/a                |         |
| `erf`                  | :white_check_mark: | :white_check_mark: |         |
| `erfc`                 | :x:                | :x:                | Not directly available in Graphcore Poplibs        |
| `erf_inv`              | :x:                | :x:                | Not directly available in Graphcore Poplibs        |
| `exp`                  | :white_check_mark: | :white_check_mark: |         |
| `expand_dims`          | :white_check_mark: | :white_check_mark: |         |
| `expm1`                | :white_check_mark: | :white_check_mark: |         |
| `fft`                  | :x:                | n/a                |         |
| `floor`                | :white_check_mark: | :white_check_mark: |         |
| `full`                 | :question:         | n/a                |         |
| `full_like`            | :question:         | n/a                |         |
| `gather`               | :white_check_mark: | n/a                | Limited set of configurations. See below. |
| `ge`                   | :white_check_mark: | n/a                |         |
| `gt`                   | :white_check_mark: | n/a                |         |
| `igamma`               | :x:                | :x:                |         |
| `igammac`              | :x:                | :x:                |         |
| `imag`                 | :x:                | :x:                |         |
| `index_in_dim`         | :x:                | n/a                |         |
| `index_take`           | :x:                | n/a                |         |
| `iota`                 | :white_check_mark: | n/a                |         |
| `is_finite`            | :white_check_mark: | n/a                |         |
| `le`                   | :white_check_mark: | n/a                |         |
| `lt`                   | :white_check_mark: | n/a                |         |
| `lgamma`               | :x:                | :x:                |         |
| `log`                  | :white_check_mark: | :white_check_mark: |         |
| `log1p`                | :white_check_mark: | :white_check_mark: |         |
| `logistic`             | :x:                | :x:                |         |
| `max`                  | :white_check_mark: | :white_check_mark: |         |
| `min`                  | :white_check_mark: | :white_check_mark: |         |
| `mul`                  | :white_check_mark: | :white_check_mark: |         |
| `ne`                   | :white_check_mark: | n/a                |         |
| `neg`                  | :white_check_mark: | :white_check_mark: |         |
| `nextafter`            | :x:                | :x:                |         |
| `pad`                  | :x:                | :x:                |         |
| `polygamma`            | :x:                | :x:                |         |
| `pow`                  | :white_check_mark: | :white_check_mark: |         |
| `real`                 | :x:                | n/a                | Complex not supported in IPU XLA backend        |
| `reciprocal`           | :white_check_mark: | :x:                |         |
| `reduce`               | :white_check_mark: | n/a                |         |
| `reshape`              | :white_check_mark: | :white_check_mark: | `dimensions` argument not supported.        |
| `rem`                  | :white_check_mark: | :white_check_mark: |         |
| `rev`                  | :x:                | :x:                |         |
| `round`                | :white_check_mark: | :white_check_mark: |         |
| `rsqrt`                | :white_check_mark: | :white_check_mark: |         |
| `scatter`              | :white_check_mark: | n/a                | Limited set of configurations. See below. |
| `scatter_add`          | :white_check_mark: | n/a                | Limited set of configurations. See below. |
| `scatter_max`          | :white_check_mark: | n/a                | Limited set of configurations. See below. |
| `scatter_min`          | :white_check_mark: | n/a                | Limited set of configurations. See below. |
| `scatter_mul`          | :white_check_mark: | n/a                | Limited set of configurations. See below. |
| `select`               | :white_check_mark: | :x:                |         |
| `shift_left`           | :white_check_mark: | :x:                |         |
| `shift_right_arithmetic`| :white_check_mark: | :x:                |         |
| `shift_right_logical`  | :white_check_mark: | :x:                |         |
| `slice`                | :white_check_mark: | n/a                |         |
| `slice_in_dim`         | :white_check_mark: | n/a                |         |
| `sign`                 | :white_check_mark: | white_check_mark   |         |
| `sin`                  | :white_check_mark: | :white_check_mark: |         |
| `sinh`                 | :x:                | :x:                |         |
| `sort`                 | :x:                | :x:                |         |
| `sort_key_val`         | :x:                | :x:                |         |
| `sqrt`                 | :white_check_mark: | :white_check_mark: |         |
| `square`               | :white_check_mark: | :x:                |         |
| `squeeze`              | :white_check_mark: | :white_check_mark: |         |
| `sub`                  | :white_check_mark: | :white_check_mark: |         |
| `tan`                  | :white_check_mark: | :white_check_mark: |         |
| `tie_in`               | :x:                | n/a                | Deprecated in JAX        |
| `top_k`                | :x:                | n/a                |         |
| `transpose`            | :white_check_mark: | :x:                | Copies the input tensor    |
| `zeta`                 | :x:                | :x:                |         |

### Limitations

* `dot_general`: Only supporting `16x16` right hand side at the moment.
* `gather/scatter`: Only first axis gather/scatter supported.

## Additional IPU optimized operations

| Operation              | Supported          | Inplace            | Remarks |
| ---------------------- | ------------------ | ------------------ |-------- |
| `scaled_add`               | :white_check_mark: | :x:                | Fused scaled add        |
| `scaled_sub`               | :white_check_mark: | :x:                | Fused scaled sub        |
