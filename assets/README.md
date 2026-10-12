# Paper figures and demo

| File | Content |
| --- | --- |
| `fig2.pipeline.jpg` | Pipeline figure from the Cube-Splat paper, displayed in the main README |
| `demo.gif` | Looping GIF of the complete demo, embedded in the main README |
| `demo.mp4` | Source video used to generate the embedded GIF |

The demo is also available on [YouTube](https://www.youtube.com/watch?v=7ZAjwrDWoHw).

To update the figure or demo, replace the corresponding file. If you change a
filename, update the references in the main README as well.

The GIF is generated from the complete MP4 at 800 pixels wide and 10 fps:

```bash
ffmpeg -i assets/demo.mp4 \
  -filter_complex 'fps=10,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3' \
  -loop 0 assets/demo.gif
```
