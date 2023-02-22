"""Performance of ODA 2D on example images.


Measure the duration and accuracy of the ODA program on images facing 
reductions in spatial resolution due to binning (1) or guassian blurring
(2) and contrast reductions due to added Poisson noise. 1 and 2 inform
us of the requirements for resolution while 3 informs us of the requirements
for contrast of the image.

We define 1 function to process the relevant variables from the Eye
object and three functions for modifying the image.

With these functions, we run through each image in this folder and perform 
the following:

1. make a new folder with the basename of the file
2. for each degrading function (reduce_size, gaussian_blur, change_contrast):
    a. make new folder with same name as function
    b. save or load a pandas dataframe for storing the following results
    c. apply function to image at 5 intensities
    d. for each intensity:
        i.   save output as {filename}_{intensity}.png and mask as same in masks
             subfolder (only really matters for the size reduction)
        ii.  make an Eye instance using output and mask
        iii. measure the following results of running ODA on the image: (1) 
             mean and s.d. of ommatidial diameter, (2) ommatidial count, and
             (3) mean +/- s.d. of running time of each key functional step
             (get_eye_outline, get_eye_dimensions, get_ommatidia, measure_ommatidia)
             and full algorithm for rough comparisons to hand counting
        iv.  Store the results from iii. into the pandas dataframe and save
3. ./benchmark_plot.py will plot and analyze the many pandas dataframes
...
"""
import sys
# sys.path.append("h:\\My Drive\\backup\\Desktop\\programs\\eye_tools\\src\\eye_tools")
# sys.path = sys.path[1:]
# import analysis_tools as et
import ODA as et
# import eye_tools as et

from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy import ndimage
import cProfile


def reduce_size(img, bin_width=1):
    """Reduce the image size and resolution by binning.


    Parameters
    ----------
    img : np.ndarray, shape=(height, width)
        The image to reduce by bin averaging.
    bin_width : int, default=1
        The pixel length of the bins used for averaging.

    Returns
    -------
    img_binned : np.ndarray, shape=(height/bin_width, width/bin_width)
        The image after bin averaging.
    """
    # round height and width to nearest whole number multiple of bin_width
    height, width = img.shape[:2]
    height_new, width_new = height//bin_width, width//bin_width
    height_scale, width_scale = int(height / height_new), int(width / width_new)
    height_clipped, width_clipped = height_new * bin_width, width_new * bin_width
    # crop img using width and height approximations
    img = np.copy(img[:height_clipped, :width_clipped])
    # resize using bin width on two new axes
    assert img.size == height_new * width_new * bin_width **2, (
        "error in resizing the array")
    new_img = np.reshape(img, newshape=(height_new, bin_width, width_new, bin_width))
    # average along the two new axes
    img_binned = new_img.mean((1, 3))
    return img_binned


def gaussian_blur(img, std=1):
    """Blur the image by convolving with a 2D gaussian.


    Parameters
    ----------
    img : np.ndarray, shape=(height, width)
        The image to blur via 2D convolution.
    std : float, default=1
        The standard deviation of the 2D gaussian kernal for convolution.
    
    Returns
    -------
    img_blurred : np.ndarray, shape=(height, width)
        The blurred image.
    """
    # to speed up computation, let's use Fourier Transform
    fft = np.fft.fft2(img)
    # apply gaussian multiplication (faster)
    ndimage.fourier_gaussian(fft, sigma=std, output=fft)
    # recover the image using inverse Fourier Transform
    img_blurred = np.fft.ifft2(fft).real
    return img_blurred


def change_contrast(img, gain=1):
    """Change the image contrast.


    Parameters
    ----------
    img : np.ndarray, shape=(height, width)
        The image to modify.
    gain : float, default=1
        The gain factor of the image after changing its contrast.

    Returns
    -------
    new_img : np.ndarray, shape=(height, width)
        The image after applying gain.
    """
    # apply gain
    dtype = img.dtype
    new_img = np.copy(img).astype(float)
    mean_val = new_img.mean()
    new_img -= mean_val
    new_img *= gain
    new_img += mean_val
    # apply shot noise to each pixel, meaning pseudo-randomly draw the new value
    # from a Poisson distribution with a mean=variance=pixel value.
    # new_img = np.random.poisson(img)
    return new_img.astype(dtype)


def process_folder(folder, img_extension='.png', regular=True, bright_peak=True):
    """Benchmark the ODA on all images of a given folder.


    Run through all images in the given folder, make a new directory, 
    apply filters to the image, and benchmark the performance of the ODA
    for each filtered image, compiling the results into a spreadsheet per 
    filtering function.

    Parameters
    ----------
    folder : path
        The path to folder containing the images to benchmarked.
    img_extension : str, default='.png'
        The file extension of the images to processed. This allows us to ignore
        irrelevant image files.
    """
    # if pixel data stored, load this
    pixel_fn = os.path.join(folder, "pixel.txt")
    if os.path.exists(pixel_fn):
        with open(pixel_fn, 'r') as txt_file:
            pixel_size = float(txt_file.read())
    else:
        pixel_size = 1
    sub_fns = os.listdir(folder)
    sub_fns = [os.path.join(folder, fn) for fn in sub_fns]
    image_fns = [fn for fn in sub_fns if fn.endswith(img_extension)]
    # offer to load a spreadsheet instead
    for image_fn in image_fns:
        if True:
            print(image_fn)
            # 1. make a new folder with the basename of the file and load image
            dirname = ".".join(image_fn.split(".")[:-1])
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            img_original = et.Layer(image_fn).load()
            if img_original.ndim > 2:
                img_original = et.rgb_2_gray(img_original)
            # check in masks folder for filename containing the image_fn
            mask_fns = os.listdir(os.path.join(folder, "masks"))
            img_fn_title = os.path.basename(image_fn).split(".")[0]
            mask_fns = [fn for fn in mask_fns if img_fn_title in fn]
            assert len(mask_fns) > 0, f"Failed to find a mask image for {image_fn}"
            mask_fn = mask_fns[0]
            mask_fn = os.path.join(folder, "masks", mask_fn)
            mask_original = et.Layer(mask_fn).load()
            # 2. for each degrading function:
            for noise_func, intensities, change_mask in zip(
                    [reduce_size, gaussian_blur, change_contrast],
                    [[1, 2,   4,   8,   16,   32  ],
                     [1, 2,   4,   8,   16,   32  ],
                     [1, 1/2, 1/4, 1/8, 1/16, 1/32][::-1]],  # keep it increasing
                    [True, False, False]):
            #   a. make new folder with same name as function
                subdirname = os.path.join(dirname, noise_func.__name__)
                if not os.path.isdir(subdirname):
                    os.mkdir(subdirname)
                for subsubdir in ["ommatidia", "reciprocal", "stats", "filtered"]:
                    full_path = os.path.join(subdirname, subsubdir)
                    if not os.path.isdir(full_path):
                        os.mkdir(full_path)
            #   b. save or load a pandas dataframe for storing the following results
                measurements = ["filename", "intensity", "lens_diameter",
                                "lens_count", "total_dur", "get_outline_dur",
                                "get_eye_dimensions_dur", "get_ommatidia_dur",
                                "measure_ommatidia_dur"]
                data_fn = os.path.join(subdirname, "data.csv")
                arr = np.zeros((len(intensities), len(measurements)))
                data = pd.DataFrame(arr, columns=measurements)
            #   c. for each intensity:
                for intensity, (num, row) in zip(intensities, data.iterrows()):
                    row.intensity = intensity
            #       i.   save output as {filename}_{intensity}.png and mask as same in masks
            #            subfolder (only really matters for the size reduction)
                    img = noise_func(img_original, intensity)
                    if img.max() > 255:
                        img = np.clip(img, 0, 255)
            #       ii.  make an Eye instance using output and mask
                    if change_mask:
                        mask = noise_func(mask_original, intensity)
                        pixel_size_alt = pixel_size * intensity  # resize
                    else:
                        mask = mask_original
                        pixel_size_alt = pixel_size
                    eye = et.Eye(arr=img, mask_arr=mask, pixel_size=pixel_size_alt)
                    eye.image = eye.image.astype('uint8')
            #       iii.   Store the following images for demonstration purposes:
            #            -eye.image
            #            -eye.reciprocal
            #            -eye.filtered_image
            #            -eye.original_image with ommatidia superimposed
                    intensity_str = f"intensity_{intensity:.3f}"
                    fn_input = os.path.join(subdirname, f"{intensity_str}.png")
                    fn_reciprocal = os.path.join(subdirname, "reciprocal", f"{intensity_str}.png")
                    fn_filtered = os.path.join(subdirname, "filtered", f"{intensity_str}.png")
                    fn_ommatidia = os.path.join(subdirname, "ommatidia", f"{intensity_str}.svg")
                    fn_stats = os.path.join(subdirname, "stats", intensity_str)
                    # store input image
                    et.save_image(fn_input, np.round(img).astype('uint8'))
                    row.filename = fn_input
                    # run profiler on oda by storing the image and mask as temporary files
                    profiler = cProfile.Profile()
                    profiler.enable()
                    eye.oda(bright_peak=bright_peak, high_pass=True, plot_fn=fn_ommatidia,
                            regular=regular, plot=False, square_lattice=False)
                    profiler.disable()
                    profiler.dump_stats(fn_stats)
                    # if noise_func == change_contrast and intensity < 1:
                    #     breakpoint()
                    # store reciprocal image
                    reciprocal = np.copy(eye.reciprocal).astype(float)
                    if not regular:
                        reciprocal = np.log(reciprocal)
                    # if noise_func == gaussian_blur:
                    #     breakpoint()
                    # if num == 0:
                    max_val = reciprocal.max()
                    reciprocal /= max_val
                    # reciprocal = (reciprocal - min_val)/max_val
                    reciprocal *= 255
                    et.save_image(fn_reciprocal, reciprocal.astype('uint8'))
                    # et.save_image(fn_reciprocal, np.log(reciprocal))
                    if len(eye.ommatidia) > 0:
                        # store filtered image
                        filtered_image = np.copy(eye.filtered_image).astype(float)
                    else:
                        filtered_image = np.zeros(eye.filtered_image.shape).astype(float)
                        filtered_image[:] = eye.filtered_image.mean()
                    filtered_image -= filtered_image.min()
                    filtered_image /= filtered_image.max()
                    filtered_image *= 255
                    et.save_image(fn_filtered, np.round(filtered_image).astype('uint8'))
            #       iv. measure the following results of running ODA on the image: (1) 
            #            mean and s.d. of ommatidial diameter, (2) ommatidial count, and
            #            (3) mean +/- s.d. of running time of each key functional step
            #            (get_eye_outline, get_eye_dimensions, get_ommatidia, measure_ommatidia)
            #            and full algorithm for rough comparisons to hand counting
                    # grab the cumulative time for the following Eye methods:
                    # 1) ommatidia_detecting_algorithm, 2) get_eye_outline,
                    # 3) get_eye_dimensions, 4) get_ommatidia, 5) measure_ommatidia
                    for method, var in zip(
                            ['ommatidia_detecting_algorithm', 'get_eye_outline',
                             'get_eye_dimensions', 'get_ommatidia', 'measure_ommatidia'],
                            ['total_dur', 'get_outline_dur', 'get_eye_dimensions_dur',
                             'get_ommatidia_dur', 'measure_ommatidia_dur']):
                        ind = [key for key in profiler.stats.keys() if key[2] == method][0]
                        stats = profiler.stats[ind]
                        row[var] = stats[3]
            #       v.  Store ommatidia results into the pandas dataframe and save
                    row['lens_diameter'] = eye.ommatidial_diameter
                    row['lens_count'] = len(eye.ommatidia)
                    data.loc[num] = row
                    data.to_csv(data_fn, index=False)


IMG_EXTENSION = ".tif"      # process only images with this filename extension
fns = os.listdir("./")
# folders = [fn for fn in fns if os.path.isdir(fn)]
folders = ['D_melanogaster_micrographs', 'ant_replicas',
           'D_mauritiana_SEM', 'D_melanogaster_SEM']
regulars = [True, False, True, True]
bright_peaks = [False, True, True, True]
# for each folder in the current directory:
for folder, regular, bright_peak in zip(folders[:1], regulars[:1], bright_peaks[:1]):
    # process each image in that folder
    print(f"Processing: {folder}")
    process_folder(folder, IMG_EXTENSION, regular, bright_peak)
