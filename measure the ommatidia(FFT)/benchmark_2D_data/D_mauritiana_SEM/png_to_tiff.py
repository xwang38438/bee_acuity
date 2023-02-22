import os
import subprocess
from PIL import Image

fns = os.listdir("./")
png_fns = [fn for fn in fns if fn.endswith(".png")]

for fn in png_fns:
    tiff_fn = fn.replace(".png", ".tif")
    if not (os.path.exists(tiff_fn)):
        img = Image.open(fn)
        img.save(tiff_fn)
        # delete png file
        os.remove(fn)
        print(tiff_fn)
