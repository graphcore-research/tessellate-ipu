# TessellateIPU supported operations

## [JAX LAX operations](https://jax.readthedocs.io/en/latest/jax.lax.html)

**Note:** Inplace operations are represented by additional primitives (e.g. `add_inplace_p`). For binary operations, only the first argument is supported for inplace update.

| Operation              | Supported          | Inplace            | Remarks |
| ---------------------- | ------------------ | ------------------ |-------- |
| `abs`                  | :white_check_mark: | :white_check_mark: |         |
| `add`                  | :white_check_mark: | :white_check_mark: |         |
| `acos`                 | :white_check_mark: | :white_check_mark: |         |
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
| `bitcase_convert_type` | :white_check_mark: | :question:         |         |
| `bitwise_not`          | :white_check_mark: | :x:                |         |
| `bitwise_and`          | :white_check_mark: | :x:                |         |
| `bitwise_or`           | :white_check_mark: | :x:                |         |
| `bitwise_xor`          | :white_check_mark: | :x:                |         |
| `population_count`     | :x:                | :x:                |         |
| `broadcast`            | :x:                | :x:                |         |
| `broadcast_in_dim`     | :x:                | :x:                |         |
| `cbrt`                 | :white_check_mark: | :white_check_mark: |         |
| `ceil`                 | :white_check_mark: | :white_check_mark: |         |
| `clamp`                | :x:                | :x:                |         |
| `collapse`             | :x:                | :x:                |         |
| `complex`              | :x:                | :x:                |         |
| `concatenate`          | :white_check_mark: | :x:                |         |
| `conj`                 | :x:                | :x:                |         |
| `conv`                 | :x:                | :x:                |         |
| `convert_element_type` | :x:                | :x:                |         |
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
| `erf`                  | :x:                | :x:                |         |
| `erfc`                 | :x:                | :x:                |         |
| `erf_inv`              | :x:                | :x:                |         |
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
| `iota`                 | :x:                | :x:                |         |
| `is_finite`            | :white_check_mark: | :x:                |         |
| `le`                   | :white_check_mark: | :x:                |         |
| `lt`                   | :white_check_mark: | :x:                |         |
| `lgamma`               | :x:                | :x:                |         |
| `log`                  | :white_check_mark: | :x:                |         |
| `log1p`                | :white_check_mark: | :x:                |         |
| `logistic`             | :x:                | :x:                |         |
| `max`                  | :white_check_mark: | :x:                |         |
| `min`                  | :white_check_mark: | :x:                |         |
| `mul`                  | :white_check_mark: | :x:                |         |
| `ne`                   | :white_check_mark: | :x:                |         |
| `neg`                  | :white_check_mark: | :x:                |         |
| `nextafter`            | :x:                | :x:                |         |
| `pad`                  | :x:                | :x:                |         |
| `polygamma`            | :x:                | :x:                |         |
| `pow`                  | :white_check_mark: | :x:                |         |
| `real`                 | :x:                | :x:                |         |
| `reciprocal`           | :x:                | :x:                |         |
| `reduce`               | :white_check_mark: | :x:                |         |
| `reshape`              | :white_check_mark: | :x:                |         |
| `rem`                  | :white_check_mark: | :x:                |         |
| `rev`                  | :white_check_mark: | :x:                |         |
| `round`                | :white_check_mark: | :x:                |         |
| `rsqrt`                | :white_check_mark: | :x:                |         |
| `scatter`              | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_add`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_max`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_min`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `scatter_mul`          | :white_check_mark: | :x:                | Limited set of configurations. See below. |
| `select`               | :x:                | :x:                |         |
| `shift_left`           | :white_check_mark: | :x:                |         |
| `shift_right_arithmetic`| :white_check_mark: | :x:                |         |
| `shift_right_logical`  | :white_check_mark: | :x:                |         |
| `slice`                | :white_check_mark: | :x:                |         |
| `slice_in_dim`         | :white_check_mark: | :x:                |         |
| `sign`                 | :white_check_mark: | :x:                |         |
| `sin`                  | :white_check_mark: | :x:                |         |
| `sinh`                 | :white_check_mark: | :x:                |         |
| `sort`                 | :x:                | :x:                |         |
| `sort_key_val`         | :x:                | :x:                |         |
| `sqrt`                 | :white_check_mark: | :x:                |         |
| `square`               | :white_check_mark: | :x:                |         |
| `squeeze`              | :white_check_mark: | :x:                |         |
| `sub`                  | :white_check_mark: | :x:                |         |
| `tan`                  | :white_check_mark: | :x:                |         |
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
