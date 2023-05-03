import jax

# Just make sure JAX can attach IPU hardware.
print(jax.devices("ipu"))
