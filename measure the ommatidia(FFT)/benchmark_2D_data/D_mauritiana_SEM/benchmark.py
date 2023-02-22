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
2. for each degrading function (reduce_size, gaussian_blur, shot_noise):
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
import eye_tools as et
import numpy as np
from scipy import ndimage


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
    height, width = img.shape
    height, width = height/bin_width, width/bin_width
    # crop img using width and height approximations
    img = np.copy(img[:height, :width])
    # resize using bin width on two new axes
    img = np.reshape(img[np.newaxis, np.newaxis],
                     newshape=(bin_width, bin_width, height/bin_width, width/bin_width))
    # average along the two new axes
    img_binned = img.mean((0, 1))
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


def shot_noise(img, gain=1):
    """Add shot noise to img after reducing image gain.


    Parameters
    ----------
    img : np.ndarray, shape=(height, width)
        The image to blur via 2D convolution.
    gain : float, default=1
        The gain factor of the image before applying shot noise. A gain of 1
        keeps the mean values for the Poisson function unchanged.

    Returns
    -------
    img_noisy : np.ndarray, shape=(height, width)
        The image after reducing its gain and applying shot noise.
    """
    # apply gain
    dtype = img.dtype
    img = (gain * np.copy(img).astype(float)).astype(dtype)
    # apply shot noise to each pixel, meaning pseudo-randomly draw the new value
    # from a Poisson distribution with a mean=variance=pixel value.
    img_noisy = np.random.poisson(img)
    return img_noisy


# todo: test the three functions

