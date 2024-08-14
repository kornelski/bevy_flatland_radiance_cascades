# Flatland Radiance Cascades

Simulates lights in 2D scenes, with soft shadows. It's a very unpolished implementation in Rust/Bevy.

It's based on the [radiance cascades paper][paper], but with a much improved storage layout:

 - It uses _pre-averaging_, so that results of tracing of individual rays don't need to be stored, and cascades store only merged average of a cone needed for the next cascade.
 - It uses _direction-first_ storage to be able to merge probes using hardware filtering (the hw part is not implemented yet).

See `src/cascades.rs` and `assets/radiance_cascades.wgsl`.

This implementation is different than most other examples:

 - It uses a "gear fix" workaround to reduce halos at the seams between cascades (instead of "bilinear fix").
 - It uses a 2x scaling factor for the lenghts of cascades and the number of rays (most use much more expensive 4x scaling).
 - It doesn't use SDFs (it naively marches density map).
 - it doesn't use mipmaps (yet).
 - There's no bounce light implemented yet.

## Links

* [what are Radiance Cascades anyway?](https://www.youtube.com/watch?v=3so7xdZHKxw)
* [the paper][paper]
* [shadertoy impl](https://www.shadertoy.com/view/mtlBzX)
* [geometry debug](https://tmpvar.com/poc/radiance-cascades/)
* [vis for v2 improvements](https://www.shadertoy.com/view/4clcWn)

[paper]: https://drive.google.com/file/d/1L6v1_7HY2X-LV3Ofb6oyTIxgEaP4LOI6/view?usp=sharing
