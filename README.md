# SpatialVAEs

Encoding and decoding of images with spatial representation of objects, i.e. objects moving around in the image.
The encoder maps the high-dimensional sparse representation ("pixels") to a low-dimensional dense representation ("cartesian coordinates"). The decoder maps the low-dimensional dense representation back to the high-dimensional sparse representation.

- `python -u train_vae.py`: Train encoder and decoder using a Variational Autoencoder (VAE)
- `python -u train_ae.py`: Train encoder and decoder using a deterministic Autoencoder (VAE)
- `python -u train_regressor`: Train a regressor (encoder) for mapping images to low-dimensional representation
- `python -u train_renderer`: Train a renderer (decocer) for mapping cartesian coordinates to images
