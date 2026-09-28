# Changelog

## 1.0.1 (2026-09-28)

- The images of the README also show on PyPI.

## 1.0.0 (2026-09-28)

The first release of torchtransform. It brings together two earlier implementations: a transform
engine that fuses chains of matrices, warps and crops into one resampling, and a large collection of
batched augmentations.

### Engine

- Geometric transforms are resolved in a single resampling of only the output pixels: an index
  gather for elements that are only flipped, turned, transposed and moved by whole pixels, one
  affine or projective grid, or a warp grid evaluated every few pixels and upsampled.
- Shrinking by two or more samples from a mipmap level per element.
- Color transforms that are affine per pixel are fused into one matrix per element and one pass.
- Every batch element draws its own parameters, and decides on its own whether a transform with a
  probability, or a branch of `OneOf` or `SomeOf`, applies to it.
- Images, masks, points, boxes and rotated boxes are transformed together, and geometry is moved
  exactly, in float64.
- 2D and 3D data.
- Seeds per call and per transform, so draws do not depend on the rest of a pipeline; `Inverse`
  undoes exactly what a transform drew in the same call.
- Replays apply the geometry of a call to other data afterward, or undo it.

### Transforms

- Composition: `Compose`, `Maybe`, `OneOf`, `SomeOf`, `Inverse`, `Identity` and `Lambda`.
- Geometric: `Affine`, `Rotate`, `RandomOrientation`, `Scale`, `Translate`, `Shear`, `Perspective`,
  `Flip`, `HorizontalFlip`, `VerticalFlip`, `QuarterTurn`, `Transpose`, `Symmetry` and
  `AffineWithinBounds`.
- Warps: `ElasticWarp`, `GridDistortion`, `LensDistortion`, `Twirl` and `Wave`.
- Canvas: `Crop`, `CenterCrop`, `RandomCrop`, `Pad`, `Resize` and `RandomResizedCrop`.
- Color: `Brightness`, `Contrast`, `Saturation`, `Hue`, `ColorJitter`, `Grayscale`, `Invert`,
  `RGBShift`, `ChannelShuffle`, `ChannelDropout`, `Sepia`, `AutoContrast`, `Normalize` and
  `ColorMatrix`.
- Photometric: `Gamma`, `Solarize`, `Posterize`, `Equalize`, `Sharpen`, `GaussianBlur`,
  `MotionBlur`, `Defocus`, `GaussianNoise`, `PoissonNoise`, `SpeckleNoise`, `SaltAndPepperNoise`,
  `JpegCompression`, `Downscale`, `Pixelate`, `Erasing`, `Vignette`, `ChromaticAberration`,
  `BackgroundGradient`, `BackgroundIrregularities` and `Clamp`.
- Documents: `BleedThrough`, `PaperTint`, `HalftoneDither`, `LowInkStreaks`, `RollerStreaks`,
  `InkSpread`, `InkErosion`, `LineFragmentation`, `EdgeShadow`, `ChannelMisalignment` and
  `BitonalCopy`.
