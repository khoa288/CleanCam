# Deposit assets and viewport-overlay workflow

The synthetic generator uses 11 RGBA deposit assets. Each asset contains an
RGB appearance layer and an alpha mask. The assets are partitioned into
disjoint train, validation, and test pools; every generated image records the
asset identifier, real parent image, transformation parameters, opacity,
blockage estimate, random seed, and inherited split.

The names and preparation categories are recorded in
[`deposit_assets.csv`](deposit_assets.csv). “Author-prepared RGBA asset” is
used where the available source record establishes the prepared asset but does
not distinguish a direct crop from other author preparation. “Procedural
Blender asset” identifies the variants explicitly produced through the shader
workflow below.

## Blender asset creation

Organic procedural assets were created with a custom Blender shader network:

1. Generated texture coordinates were mixed at factor 0.1 with a three-
   dimensional fBm Noise Texture at scale 18.0, breaking spatial linearity.
2. The warped coordinates drove a two-dimensional Hybrid Multifractal Noise
   Texture. A Cardinal-interpolated Color Ramp thresholded its output into the
   alpha morphology. Deposit size and density were varied through the noise
   scale and threshold position; the base settings were 4.7 and 0.160.
3. A second, Linear-interpolated Color Ramp mapped the warped coordinates to a
   green pigment gradient used as Base Color.
4. A Principled BSDF represented wet biomass with Roughness 1.0, Transmission
   Weight 0.719, and Subsurface Weight 0.215.

The node layout is retained in
`documentation/blender_shader_network.jpeg` in the data release and
`docs/manuscript_sources/blender_shader_network.jpeg` in the code repository.

## Image compositing

For a real parent image of width \(W\) and height \(H\), an asset is resized to
cover 1.1 times the image diagonal, randomly rotated, centre-cropped, and
downsampled then upsampled by an integer factor \(d \in [15,35]\). The
effective opacity at pixel \((x,y)\) is

\[
\alpha(x,y) = o\,\widetilde{M}(x,y)/255,
\]

where \(o \sim U(0.10,0.60)\) and \(\widetilde{M}\) is the transformed alpha
mask. The mean obstruction is

\[
b = \frac{1}{WH}\sum_{x=1}^{W}\sum_{y=1}^{H}\alpha(x,y).
\]

The RGB synthetic image is then

\[
I_{\mathrm{syn}}(x,y,c) =
\operatorname{clip}\!\left(
I_{\mathrm{parent}}(x,y,c) +
\alpha(x,y)\left[\widetilde{A}(x,y,c)-I_{\mathrm{parent}}(x,y,c)\right],
0,255\right).
\]

The generator maps \(b\) to an ordinal label increment using the frozen
thresholds in `scripts/build_cleancam_release.py`, rejects outputs that do not
reach the requested target label, and saves accepted images as JPEG at quality
95.
