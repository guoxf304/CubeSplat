# Depth inference component provenance

The inference runtime includes the UniFuse network files from
[UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion) and an
ERP-to-cubemap sampler extracted from
[Depth Anywhere](https://github.com/albert100121/Depth-Anywhere). The sampler
credits [py360convert](https://github.com/sunset1995/py360convert).

The upstream license texts are preserved here:

- `UniFuse.txt`: MIT license for UniFuse.
- `Depth-Anywhere.txt`: Apache License 2.0 for Depth Anywhere.
- `py360convert.txt`: MIT license for py360convert.

Cube-Splat includes an inference-only packaging of the sampler to avoid importing
unrelated training models. The UniFuse runtime has a NumPy boolean-alias
compatibility change; the local encoder cache directory is retained. The loader
uses the full checkpoint without separately downloading an ImageNet encoder and
caches the loaded model between keyframes. Model checkpoint files are distributed
separately and are not covered by a new license in this repository.
