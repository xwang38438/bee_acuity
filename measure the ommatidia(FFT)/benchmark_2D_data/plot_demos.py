"""Plot the ant eye comparisons and one D. melanogaster with ommatidial centers.

The ant eye comparison
"""

#!/usr/bin/env python

"""Batch process all images of compound eyes and save data.


Assumes the following folder structure:

.\
|--analysis.py
|--image_001.jpg
|--image_002.jpg
|...
|--masks\
   |--image_001.jpg             # white sillouetting mask on black background
   |--image_002.jpg
   |...
|--demos\
   |--image_001.jpg (outcome)
   |--image_002.jpg (outcome)
   |...
|--eye_data.csv     (outcome)
|--_hidden_file
"""
import os
from scipy import misc
from ODA import *
# from eye_tools import *
import pandas as pd


# Custom parameters
# if a list, must be one-to-one with images
REGULAR = False                 # True assumes the lattice is regular
BRIGHT_PEAK = True             # True assumes a bright point for every peak
HIGH_PASS = True               # True adds a high-pass filter to the low-pass used in the ODA
SQUARE_LATTICE = False          # True assumes only two fundamental gratings
IMG_EXTENSION = ".tif"        # assumes you're only interested in this file extension

def process_folder(folder, img_extension=IMG_EXTENSION, regular=True, bright_peak=True, min_val=10, max_val=20):
    # if pixel data stored, load this
    pixel_fn = os.path.join(folder, "pixel.txt")
    if os.path.exists(pixel_fn):
        with open(pixel_fn, 'r') as txt_file:
            pixel_size = float(txt_file.read())
    else:
        pixel_size = 1
    sub_fns = os.listdir(folder)
    sub_fns = [os.path.join(folder, fn) for fn in sub_fns]
    image_fns = [fn for fn in sub_fns if fn.endswith(IMG_EXTENSION)]
    # offer to load a spreadsheet instead
    eyes = []
    # apply the ODA-2D to each image and store the output
    for image_fn in image_fns:
        img_fn_title = os.path.basename(image_fn).split(".")[0]
        print(image_fn)
        img_original = Layer(image_fn).load()
        if img_original.ndim > 2:
            img_original = rgb_2_gray(img_original)
        # check in masks folder for filename containing the image_fn
        mask_fns = os.listdir(os.path.join(folder, "masks"))
        mask_fns = [fn for fn in mask_fns if img_fn_title+IMG_EXTENSION == fn]
        assert len(mask_fns) > 0, f"Failed to find a mask image for {image_fn}"
        mask_fn = mask_fns[0]
        mask_fn = os.path.join(folder, "masks", mask_fn)
        mask_original = Layer(mask_fn).load()
        # make an eye object and get the ommatidial centers
        eye = Eye(arr=img_original, mask_arr=mask_original, pixel_size=pixel_size)
        eye.image = eye.image.astype('uint8')
        eye.crop_eye()
        eye.oda(bright_peak=bright_peak, high_pass=True, regular=regular, plot=False, square_lattice=False,
                manual_edit=True)
        eyes += [eye]
    # make a figure with 2 rows, 5 columns. top row is all of the cropped images
    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(15.5, 6))
    # sort the eye objects by ommatidial number
    counts = [len(eye.ommatidial_inds) for eye in eyes]
    order = np.argsort(counts)
    eyes = np.array(eyes)
    # for each column, plot the image on top and the image with ommatidia superimposed on the bottom
    cols = axes.T
    for col, eye in zip(cols, eyes[order]):
        # use the mask image to crop each picture to a square
        mask_ys, mask_xs = np.where(eye.mask)
        # give 10% padding to the larger range, then make equal on the other one
        width, height = mask_xs.ptp(), mask_ys.ptp()
        breakpoint()
        crop_length = int(np.round(1.05 * max([width, height])))
        center_x, center_y = mask_xs.mean(), mask_ys.mean()
        left_b, right_b = round(center_x - crop_length/2), round(center_x + crop_length/2)
        bottom_b, top_b = round(center_y - crop_length/2), round(center_y + crop_length/2)
        # plot the original image
        col[0].imshow(eye.image[bottom_b : top_b, left_b : right_b], cmap='gray')
        col[1].imshow(eye.image[bottom_b : top_b, left_b : right_b], cmap='gray', zorder=1, alpha=.5)
        # get the ommatidial coordinates
        ys, xs = eye.ommatidial_inds.T
        xs -= left_b
        ys -= bottom_b
        diams = eye.ommatidial_diameters * 1000
        no_nans = np.isnan(diams) == False
        # get diams in pixel values
        diams_pxl = diams / (pixel_size * 1000)
        diams_pxl *= .6
        # areas = diams_pxl
        areas = np.pi * (diams_pxl / 2) ** 2
        scatter = col[1].scatter(
            xs[no_nans], ys[no_nans], c=diams[no_nans], 
            vmin=min_val, vmax=max_val, cmap='viridis', 
            marker='.', edgecolors='none', zorder=2, s=areas)
    # formatting
    scale_length = .05 / pixel_size
    for row_num, row in enumerate(axes):
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for key in ['left', 'right', 'bottom', 'top']:
                ax.spines[key].set_visible(False)
            # plot a bar for scale
            # we pknow the length of an individual pixel in mm. how many pixels makes 50 um? 
            # 50 um = .05 mm, 1/pxl = x/.05 => x = pxl * .05
            ax.plot([0, scale_length], [0, 0], lw=10, color='w', zorder=2)
            ax.invert_yaxis()
    plt.tight_layout(pad=0, rect=[0, 0, .91, 1])
    # add the colorbar
    cbar_ax = fig.add_axes(rect=(.93, 0.02, .01, .43))
    fig.colorbar(scatter, cax=cbar_ax, orientation='vertical')
    cbar_ax.set_ylabel("Lens Diameter ($\mu$m)", rotation=270, labelpad=25)
    plt.savefig(f"{folder}_comparison.svg")
    plt.show()

# folders = [fn for fn in fns if os.path.isdir(fn)]
folders = ['D_melanogaster_micrographs', 'ant_replicas']
regulars = [True, False, True, True]
bright_peaks = [False, True, True, True]
# for each folder in the current directory:
ind=0
for folder, regular, bright_peak in zip(folders[ind:], regulars[ind:], bright_peaks[ind:]):
    # process each image in that folder
    print(f"Processing: {folder}")
    process_folder(folder, IMG_EXTENSION, regular=regular, bright_peak=bright_peak)