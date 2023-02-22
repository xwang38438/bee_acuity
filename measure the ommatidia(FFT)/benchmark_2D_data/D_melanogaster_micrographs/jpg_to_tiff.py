import os
import subprocess
from PIL import Image

INPUT_EXTENSION = ".jpg"
OUTPUT_EXTENSION = ".tif"

fns = os.listdir("./")
fns = [fn for fn in fns if fn.endswith(INPUT_EXTENSION)]

for fn in fns:
    new_fn = fn.replace(INPUT_EXTENSION, OUTPUT_EXTENSION)
    if not (os.path.exists(new_fn)):
        img = Image.open(fn)
        img.save(new_fn)
        # delete png file
        os.remove(fn)
        print(new_fn)
