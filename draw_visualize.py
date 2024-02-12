import numpy as np
from utils import show
from draw import Draw

rng = np.random.default_rng()
img_size = 500
draw = Draw(img_size, rng)

rows = 50
cols = 4

def show_draw_function(func, rows=1, cols=5):
    n = rows * cols
    m = cols
    imgs = []
    for _ in range(n):
        img = func()
        imgs.append(img)
    grouped_imgs = [imgs[i:i+m] for i in range(0, len(imgs), m)]
    for group in grouped_imgs:
        show(group, grayscale=True)

# show_draw_function(draw.line, rows=rows, cols=cols)
# show_draw_function(draw.rectangle_outline, rows=rows, cols=cols)
# show_draw_function(draw.ellipse, rows=rows, cols=cols)
# show_draw_function(draw.stickman, rows=rows, cols=cols)
# show_draw_function(draw.dashed_arrow, rows=rows, cols=cols)