# TessellateIPU supported operations

## [JAX LAX operations](https://jax.readthedocs.io/en/latest/jax.lax.html)

**Note:** Inplace operations are represented by additional primitives (e.g. `add_inplace_p`). For binary operations, only the first argument is supported for inplace update.

| Operation              | Supported          | Inplace            | Remarks |
| ---------------------- | ------------------ | ------------------ |-------- |
| `abs`                  | :white_check_mark: | :white_check_mark: |         |
| `add`                  | :white_check_mark: | :white_check_mark: |         |
| `acos`                 | :x:                | :x:                | Not directly available in Graphcore Poplibs        |
| `approx_max_k`         | :x:                | :x:                |         |
| `approx_min_k`         | :x:                | :x:                |         |
| `argmax`               | :x:                | :x:                |         |
| `argmin`               | :x:                | :x:                |         |
| `asin`                 | :white_check_mark: | :white_check_mark: |         |
| `atan`                 | :white_check_mark: | :white_check_mark: |         |
| `atan2`                | :white_check_mark: | :white_check_mark: |         |
| `batch_matmul`         | :x:                | :x:                | See `dot_general`         |
| `bessel_i0e`           | :x:                | :x:                |         |
| `bessel_i1e`           | :x:                | :x:                |         |
| `betainc`              | :x:                | :x:                |         |
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
| `collapse`             | :x:                | :x:                |         |
| `complex`              | :x:                | :x:                |         |
| `concatenate`          | :white_check_mark: | :x:                |         |
| `conj`                 | :x:                | :x:                |         |
| `conv`                 | :x:                | :x:                |         |
| `convert_element_type` | :white_check_mark: | :x:                |         |
| `conv_general_dilated` | :x:                | :x:                |         |
| `conv_transpose`       | :x:                | :x:                |         |
| `cos`                  | :white_check_mark: | :white_check_mark: |         |
| `cosh`                 | :white_check_mark: | :white_check_mark: |         |
| `cummax`               | :x:                | :x:                |         |
| `cummin`               | :x:                | :x:                |         |
| `cumprod`              | :x:                | :x:                |         |
| `cumsum`               | :x:                | :x:                |         |
| `digamma`              | :x:                | :x:                |         |
| `div`                  | :white_check_mark: | :white_check_mark: |         |
| `dot`                  | :white_check_mark: | :x:                |         |
| `dot_general`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `dynamic_slice`        | :x:                | :x:                |         |
| `dynamic_update_slice` | :x:                | :x:                |         |
| `eq`                   | :white_check_mark: | :x:                |         |
| `erf`                  | :white_check_mark: | :white_check_mark: |         |
| `erfc`                 | :x:                | :x:                | Not directly available in Graphcore Poplibs        |
| `erf_inv`              | :x:                | :x:                | Not directly available in Graphcore Poplibs        |
| `exp`                  | :white_check_mark: | :x:                |         |
| `expand_dims`          | :white_check_mark: | :x:                |         |
| `expm1`                | :white_check_mark: | :x:                |         |
| `fft`                  | :x:                | :x:                |         |
| `floor`                | :white_check_mark: | :x:                |         |
| `full`                 | :question:         | :x:                |         |
| `full_like`            | :question:         | :x:                |         |
| `gather`               | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `ge`                   | :white_check_mark: | :x:                |         |
| `gt`                   | :white_check_mark: | :x:                |         |
| `igamma`               | :x:                | :x:                |         |
| `igammac`              | :x:                | :x:                |         |
| `imag`                 | :x:                | :x:                |         |
| `index_in_dim`         | :x:                | :x:                |         |
| `index_take`           | :x:                | :x:                |         |
| `iota`                 | :white_check_mark: | n/a                |         |
| `is_finite`            | :white_check_mark: | :x:                |         |
| `le`                   | :white_check_mark: | :x:                |         |
| `lt`                   | :white_check_mark: | :x:                |         |
| `lgamma`               | :x:                | :x:                |         |
| `log`                  | :white_check_mark: | :white_check_mark: |         |
| `log1p`                | :white_check_mark: | :white_check_mark: |         |
| `logistic`             | :x:                | :x:                |         |
| `max`                  | :white_check_mark: | :white_check_mark: |         |
| `min`                  | :white_check_mark: | :white_check_mark: |         |
| `mul`                  | :white_check_mark: | :white_check_mark: |         |
| `ne`                   | :white_check_mark: | :x:                |         |
| `neg`                  | :white_check_mark: | :white_check_mark: |         |
| `nextafter`            | :x:                | :x:                |         |
| `pad`                  | :x:                | :x:                |         |
| `polygamma`            | :x:                | :x:                |         |
| `pow`                  | :white_check_mark: | :white_check_mark: |         |
| `real`                 | :x:                | :x:                |         |
| `reciprocal`           | :white_check_mark: | :x:                |         |
| `reduce`               | :white_check_mark: | :x:                |         |
| `reshape`              | :white_check_mark: | :white_check_mark: | `dimensions` argument not supported.        |
| `rem`                  | :white_check_mark: | :white_check_mark: |         |
| `rev`                  | :x:                | :x:                |         |
| `round`                | :white_check_mark: | :white_check_mark: |         |
| `rsqrt`                | :white_check_mark: | :white_check_mark: |         |
| `scatter`              | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_add`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_max`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_min`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_mul`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `select`               | :white_check_mark: | :x:                |         |
| `shift_left`           | :white_check_mark: | :x:                |         |
| `shift_right_arithmetic`| :white_check_mark: | :x:                |         |
| `shift_right_logical`  | :white_check_mark: | :x:                |         |
| `slice`                | :white_check_mark: | :x:                |         |
| `slice_in_dim`         | :white_check_mark: | :x:                |         |
| `sign`                 | :white_check_mark: | :x:                |         |
| `sin`                  | :white_check_mark: | :white_check_mark: |         |
| `sinh`                 | :x:                | :x:                |         |
| `sort`                 | :x:                | :x:                |         |
| `sort_key_val`         | :x:                | :x:                |         |
| `sqrt`                 | :white_check_mark: | :white_check_mark: |         |
| `square`               | :white_check_mark: | :x:                |         |
| `squeeze`              | :white_check_mark: | :white_check_mark: |         |
| `sub`                  | :white_check_mark: | :white_check_mark: |         |
| `tan`                  | :white_check_mark: | :white_check_mark: |         |
| `tie_in`               | :x:                | :x:                | Deprecated in JAX        |
| `top_k`                | :x:                | :x:                |         |
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
