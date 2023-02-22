import eye_tools as et
import os


fns = os.listdir("./")
img_fns = [fn for fn in fns if fn.endswith(".jpg")]
# for each image, 
if not os.path.isdir("masks"):
    os.mkdir("masks")
# use color selector to make masks for each image
for fn in img_fns:
    layer = et.Layer(fn)
    mask = layer.color_key()
    mask = 255 * mask.astype('uint8')
    # save in masks folder as 8 bit image
    new_fn = os.path.join("masks", fn)
    et.save_image(new_fn, mask)
