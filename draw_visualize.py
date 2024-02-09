import numpy as np
from utils import show
from draw import Draw

rng = np.random.default_rng()
img_size = 100
draw = Draw(img_size, rng)

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

show_draw_function(draw.line)
show_draw_function(draw.rectangle_outline)
show_draw_function(draw.ellipse)
show_draw_function(draw.stickman)
show_draw_function(draw.dashed_arrow)