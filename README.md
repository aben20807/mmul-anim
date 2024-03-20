Visualization of cache-optimized matrix multiplication
======================================================

The code in this repository generates animations visualizing
algorithms for cache-optimized matrix multiplication. Generated
animations are available on
[YouTube](https://www.youtube.com/playlist?list=PLB_aWiiTt1af-dICxt6E7pNJWrfcqHE2g).
One frame from the animation is shown below.

![Example](/mmul-example.png)

To generate the animations, simply run `make` (legacy, only available at `379bf8`).

Dependencies:

- python3-cairo (`pip install pycairo`)
- ffmpeg (`choco install ffmpeg` in windows)

How to run the modified version:

- The default settings only perform a dry run and print the cache hit rate
- Use `-t pdf` (fast) or `-t mp4` (slow) to generate the visualization output
- Use `--L1 0` to utilize a one-level cache instead of a two-level cache

```bash
$ python matrix_mul.py --help
usage: matrix_mul.py [-h] [--matrix-size SIZE] [--transpose] [--cache-line SIZE]
                     [--L1 SIZE] [--L2 SIZE] [--block1 SIZE] [--block2 SIZE] [--no-memory]
                     [--title TITLE] [--subtitle SUBTITLE] [--output FILENAME]
                     [--type {dry,pdf,mp4}] [--framerate FRAMERATE]

Visualization of cache-optimized matrix multiplication

options:
  -h, --help            show this help message and exit
  --matrix-size SIZE    n of the n-by-n matrix (default: 16)
  --transpose           Transpose matrix B (default: False)
  --cache-line SIZE     number of elements for each cache line, must be power of 2 (default: 4)
  --L1 SIZE             number of cache lines in L1 cache (default: 4)
  --L2 SIZE             number of cache lines in L2 cache (default: 16)
  --block1 SIZE         Inner block size (default: 16)
  --block2 SIZE         Outer block size (default: 4)
  --no-memory           Do not draw memory (default: False)
  --title TITLE
  --subtitle SUBTITLE
  --output FILENAME, -o FILENAME
                        Output file (default: matrix_mul.pdf)
  --type {dry,pdf,mp4}, -t {dry,pdf,mp4}
                        The type of the output file (default: dry)
  --framerate FRAMERATE
                        mp4 framerate (default: 24)
```
