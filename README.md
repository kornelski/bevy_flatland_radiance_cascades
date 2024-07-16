# Flatlang Radiance Cascades — very vanilla implementation

Simulates lights in 2D scenes, with soft shadows.

Unfinished, buggy. I'm a noob in shader graphics, and don't quite know what I'm doing. Based on Rust/Bevy.

 - no pre-averaging (yet)
 - no bilinear fix
 - no mipmaps
 - no SDFs even (naively marches density map)
 - no proper bounce light (can this even work without SDFs?)
 - 2x ray growth factor, not 4x, because that's what the paper promised!

See `src/cascades.rs` and `cascades_*.wgsl` shaders in `assets/`.

## Links

* [what are Radiance Cascades anyway?](https://www.youtube.com/watch?v=3so7xdZHKxw)
* [the paper](https://drive.google.com/file/d/1L6v1_7HY2X-LV3Ofb6oyTIxgEaP4LOI6/view?usp=sharing)
* [shadertoy impl](https://www.shadertoy.com/view/mtlBzX)
* [geometry debug](https://tmpvar.com/poc/radiance-cascades/)
* [vis for v2 improvements](https://www.shadertoy.com/view/4clcWn)
