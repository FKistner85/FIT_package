# Figure Captions

Generated plots store a short caption next to each image. The helper
`FIT_python.caption_utils.save_caption` writes a `.txt` file with the
same name as the figure. Plotting functions call this automatically so
`results/figures/my_plot.png` will have a matching
`results/figures/my_plot.txt`. Captions are also printed to stdout which
is useful when running notebooks.
