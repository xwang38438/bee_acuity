"""Import the results of benchmark.py.


There are several folders containing images of compound eyes. After running 
benchmark.py, each folder should have the following structure:

.\
|--dataset1\
    |--image_001.jpg                # original image
    |--image_001\                   # generated data:
        |--gaussian_blur\
            |--intensity_1.png      # images gaussian blurred
            |--intensity_1.png      # images gaussian blurred
            |--intensity_2.png
            |--...
            |--filtered\
                |--intensity_1.png  # post-ODA filtered images
                |--intensity_2.png
                |--...
            |--ommatidia\
                |--intensity_1.svg  # input images with ommatidia scatter
                |--intensity_2.svg
                |--...
            |--reciprocal\
                |--intensity_1.png  # reciprocal images used in ODA
                |--intensity_2.png
                |--...
            |--stats\
                |--intensity_1      # cProfile pstats
                |--intensity_2
                |--...
        |--reduce_size\
            |--intensity_1.png      # images bin averaged
            |--intensity_2.png
            |--...                  # same as .\reduce_size\
        |--change contrast\
            |--intensity_1.png      # images after applying gain
            |--intensity_2.png
            |--...
    |--image_002.jpg
    |--image_002\
        |--...                      # same contents as image_001
    |--...

To assess the effect of each filter, we want to accumulate all of the data from 
the numerous datasets into one spreadsheet per filter type using a wide format. 
To ensure a unique index for each image at at each filter intensity level, each 
filter dataset will have an additional column of folder and intensity level. There 
will be three spreadsheets: [reduce_size, gaussian_blur, change_contrast]. 


Outline:
0. make the following empty datasets for combining the many spreadsheets:
    a. make an empty array called data with the following shape:
        N = image number X 8 intensity levels X  7 measurements
        the intensity levels for reduce_size and gaussian_blur are:
            [1,  2,   4,   8,   16,   32]
        the intensity levels for change_contrast are:
            [1,  1/2, 1/4, 1/8, 1/16, 1/32]
        the 7 measurements are:
            [lens_diameter, lens_count, total_dur, get_outline_dur,
            get_eye_dimensions_dur, get_ommatidia_dur, measure_ommatidia_dur]
    b. for each noise function, an empty pandas dataframe called {function}_dataset.csv
    with the following columns:
        [folder, filename, intensity, lens_diameter, lens_count,
        total_dur, get_outline_dur, get_eye_dimensions_dur, get_ommatidia_dur,
        measure_ommatidia_dur].
1. For each noise function subfolder within each image folder:
    a. make a summary figure for that image showing the 4 saved images in
    8 rows for each intensity by 4 columns for each image.
    b. copy the contents of data.csv into dataset.csv, adding the folder name
    (pertaining to different species) and filename (pertaining to each image).
2. With the data ndarray and dataset.csv pandas dataframe, for each folder name, 
noise function, and measurement:
    a. plot absolute measurement by absolute intensity
    b. plot absolute measurement by relative intensity (reduce_size: pixels/diameter, 
    gaussian_blur: pixels/std, change_contrast: contrast=(max-min)/(max+min))
    c. plot relative measurement (measurement/hand measurement) by relative intensity
For each plot, plot a scatterplot with connecting lines for each individual and
low alpha color corresponding to each main folder. For the first set of plots, also 
calculate the following population statistics per intensity: 25, 50, 75th percentiles, 
the bootstrapped 99% CI for the mean, and the mean. Plot these for each intensity level,
using a low opacity line for the IQR and a full opacity line for the CI.
"""
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import statsmodels.formula.api as smf
from scipy import stats
import seaborn as sbn

sbn.set_style("ticks")
sbn.set_style({"xtick.direction": "in","ytick.direction": "in"})



blue, green, yellow, orange, red, purple = [(0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37), (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]
colors = [blue, red, green, orange, yellow, purple]

PLOT_DEMO = False

fns = os.listdir("./")
# folders = [fn for fn in fns if os.path.isdir(fn)]
folders = ['D_melanogaster_micrographs', 'ant_replicas',
           'D_mauritiana_SEM', 'D_melanogaster_SEM']
# radii = {'ant_replicas': 50, 'Tam16 D_mauritiana': 100,
#          'Zi251 D_melanogaster': 100}

noise_functions = ['reduce_size', 'change_contrast']
# 0. make the following empty datasets for combining the many spreadsheets:
#     a. make an empty array called data with the following shape:
#         N = image number X 6 intensity levels X  7 measurements
#         the intensity levels for reduce_size and gaussian_blur are:
#             [1,  2,   4,   8,   16,   32  ]
#         the intensity levels for change_contrast are:
#             [1,  1/2, 1/4, 1/8, 1/16, 1/32]
#         the 7 measurements are:
#             [lens_diameter, lens_count, total_dur, get_outline_dur,
#             get_eye_dimensions_dur, get_ommatidia_dur, measure_ommatidia_dur]
#     b. for each noise function, an empty pandas dataframe called {function}_dataset.csv
#     with the following columns:
#         [folder, filename, intensity, lens_diameter, lens_count,
#         total_dur, get_outline_dur, get_eye_dimensions_dur, get_ommatidia_dur,
#         measure_ommatidia_dur].
data = {'reduce_size': [], 'gaussian_blur': [], 'change_contrast': []}
dataset = {"function": [], "folder":[], "filename":[], "intensity":[],
           "lens_diameter":[], "lens_count":[], "total_dur":[],
           "get_outline_dur":[], "get_eye_dimensions_dur":[],
           "get_ommatidia_dur":[], "measure_ommatidia_dur":[]}
measurements = tuple(dataset.keys())[2:]
# 1. For each noise function subfolder within each image folder:
contrasts = {}
# make a summary figure for that image showing the 4 saved images in
# 6 rows for each intensity by 4 columns for each image.
# fig, axes = plt.subplots(ncols=6, nrows=4, figsize=(12,9))  # empty figure
fig, axes = plt.subplots(ncols=6, nrows=4, figsize=(12,6))  # empty figure
for folder in folders:
    # go through each folder per dataset
    img_folders = os.listdir(folder)
    img_folders = [os.path.join(folder, fn) for fn in img_folders]
    img_folders = [fn for fn in img_folders if os.path.isdir(fn)]
    for img_folder in img_folders:
        # and each folder generated per image
        # noise_folders = os.listdir(img_folder)
        noise_folders = np.copy(noise_functions)
        noise_folders = [os.path.join(img_folder, fn) for fn in noise_folders if not fn.startswith("_")]
        noise_folders = [fn for fn in noise_folders if os.path.isdir(fn)]
        for noise_folder in noise_folders:
            # and each folder with degraded versions of that image
            noise_type = os.path.basename(noise_folder)
            data_fn = os.path.join(noise_folder, "data.csv")
            if os.path.exists(data_fn):
                noise_data = pd.read_csv(data_fn)
                demo_folders = ['', 'reciprocal', 'filtered']
                demo_folders = [os.path.join(noise_folder, fn) for fn in demo_folders]
                for row_num, (demo_folder, row, cmap) in enumerate(zip(
                        demo_folders, [axes[0], axes[1], axes[3]],
                        ['gray', 'Greys', 'gray'])):
                    # plot the original images in the first column
                    img_fns = os.listdir(demo_folder)
                    img_fns = [fn for fn in img_fns if fn.endswith(".png")]
                    intensity_vals = [fn.replace(".png", "") for fn in img_fns]
                    intensity_vals = [float(val.split("_")[1]) for val in intensity_vals]
                    original_imgs = [os.path.join(demo_folder, fn) for fn in img_fns]
                    order = np.argsort(intensity_vals)
                    img_fns = np.array(original_imgs)[order]
                    intensity_vals = np.array(intensity_vals)[order]
                    for col_num, (fn, intensity, ax) in enumerate(zip(
                            img_fns, intensity_vals, row)):
                        # get contrast as the root mean squared of the image
                        contrast_fn = fn + ".contrast.txt"
                        # if os.path.exists(contrast_fn):
                        if False:
                            with open(contrast_fn, 'r') as txt_file:
                                contrast = float(txt_file.read())
                        else:
                            try:
                                img = np.asarray(Image.open(fn)).astype(float)
                            except:
                                breakpoint()
                            img /= 255
                            contrast = img.std()
                            with open(contrast_fn, 'w') as txt_file:
                                txt_file.write(str(contrast))
                        contrasts[fn] = contrast
                        if PLOT_DEMO:
                            img = np.asarray(Image.open(fn))
                            if img.mean() > 0:
                                ax.imshow(img, cmap=cmap,
                                          vmin=0, vmax=np.iinfo(img.dtype).max)
                            # formatting
                            # title:
                            if row_num == 0:
                                ax.set_title(intensity)
                            # x-axis
                            ax.set_xticks([])
                            if col_num == 0:
                                ax.set_ylabel(os.path.basename(demo_folder))
                            # y-axis
                            ax.set_yticks([])
                            # remove spines
                            for spine in ax.spines:
                                ax.spines[spine].set_visible(False)
                            if 'reciprocal' in demo_folder:
                                # plot the zoom box
                                width, height = img.shape
                                # radius = radii[folder]  # px
                                radius = 200
                                center_y, center_x = float(width)/2, float(height)/2
                                ax.plot([center_x - radius, center_x - radius, center_x + radius,
                                         center_x + radius, center_x - radius],
                                        [center_y - radius, center_y + radius, center_y + radius,
                                         center_y - radius, center_y - radius], color='k')
                                zoom_ax = axes[2][col_num]
                                zoom_ax.imshow(img, cmap='Greys',
                                               vmin=0, vmax=np.iinfo(img.dtype).max)
                                zoom_ax.set_xlim(center_x - radius, center_x + radius)
                                zoom_ax.set_ylim(center_y - radius, center_y + radius)
                                # formatting
                                # x-axis
                                zoom_ax.set_xticks([])
                                if col_num == 0:
                                    zoom_ax.set_ylabel("reciprocal\nzoomed")
                                # y-axis
                                zoom_ax.set_yticks([])
                                # remove spines
                                # for spine in zoom_ax.spines:
                                #     zoom_ax.spines[spine].set_visible(False)
                # b. copy the contents of data.csv into dataset.csv, adding the folder name
                # (pertaining to different species) and filename (pertaining to each image).
                arr = noise_data.values[:, 2:]
                data[noise_type] += [arr]
                for num, row in noise_data.iterrows():
                    noise_type = os.path.basename(os.path.split(row.filename)[0])
                    dataset['function'] += [noise_type]
                    dataset['folder'] += [folder]
                    for measurement in measurements:
                        dataset[measurement] += [row[measurement]]
            if PLOT_DEMO:
                plt.tight_layout()
                demo_fn = noise_folder + ".png"
                # plt.savefig(demo_fn, dpi=1000)
                print(demo_fn)
                for row in axes:
                    for ax in row:
                        ax.clear()
dataset = pd.DataFrame(dataset)
absolute_dataset = dataset
# to get relative measurements, we need reference measurements of ommatidial diameter
# and count. This way we can convert intensity into real units: pixels/ommatidium for
# reduce_size and gaussian_blur and contrast for change_contrast
counts_fn = "counts.csv"
reference_data = pd.read_csv(counts_fn)
reference_data['diameter'] = np.repeat(np.nan, len(reference_data))
reference_data['pixel'] = np.repeat(np.nan, len(reference_data))
for folder in folders:
    # load coordinates from running tracker.py
    data_fn = os.path.join(folder, "data.npy")
    data = np.load(data_fn)
    if data.shape[0] > 6:
        data = data[:-2]
    filenames = np.load(os.path.join(folder, 'order.npy'))
    # load pixel size to convert from pixel to distance
    pixel_fn = os.path.join(folder, "pixel.txt")
    with open(pixel_fn, 'r') as txt_file:
        pixel = float(txt_file.read())
    data *= pixel
    # calculate the 9 nearest neighbor diameters
    comparisons = [[0, 1], [0, 5], [1, 5], [1, 3], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5]]
    dists = data[np.array(comparisons)]
    dists = dists[:, 0] - dists[:, 1]
    dists = np.linalg.norm(dists, axis=2)
    diams = dists.mean(0)
    # store in reference dataframe
    for fname, diam in zip(filenames, diams):
        base = os.path.basename(fname)
        i = reference_data.file.values == base
        if any(i):
            reference_data.diameter[i] = diam
            reference_data.pixel[i] = pixel
# store the reference data into the grand dataset
diameters = []
counts = []
pixels = []
for num, row in dataset.iterrows():
    # get reference filename
    filename = row.filename
    filename = os.path.basename(os.path.dirname(os.path.dirname(filename)))
    # find index of corresponding reference data
    i = [filename in val for val in reference_data['file'].values]
    diameters += [reference_data.diameter[i].mean()]
    counts += [reference_data['count'][i].mean()]
    # fix pixel size for reduced size images
    if row.function == 'reduce_size':
        pixels += [row.intensity * reference_data['pixel'][i].mean()]
    else:
        pixels += [reference_data['pixel'][i].mean()]
dataset['reference_count'] = counts
dataset['reference_diam'] = diameters
dataset['pixel'] = pixels
# normalize intensity values:
# for those with distance parameters, normalize by the reference diameter
reduce_size = dataset.function.values == 'reduce_size'
gaussian_blur = dataset.function.values == 'gaussian_blur'
change_contrast = dataset.function.values == 'change_contrast'
# for contrast, leave alone
dataset['intensity_rel'] = dataset.intensity.copy()
# normalize the measured values:
dataset.intensity_rel[reduce_size] = (dataset.reference_diam[reduce_size]/
                                       dataset.pixel[reduce_size])
dataset.intensity_rel[gaussian_blur] = (dataset.reference_diam[gaussian_blur]/
                                        (dataset.intensity * dataset.pixel))
contrast_vals = []
for num, row in dataset.iterrows():
    if row.filename in contrasts.keys():
        contrast_vals += [contrasts[row.filename]]
    else:
        contrast_vals += [np.nan]
contrast_vals = np.array(contrast_vals)
dataset.intensity_rel[change_contrast] = contrast_vals[change_contrast]
dataset['lens_diameter_rel'] = dataset.lens_diameter / dataset.reference_diam
dataset['lens_count_rel'] = dataset.lens_count / dataset.reference_count

# 2. With the data ndarray and dataset.csv pandas dataframe, make 2 figures:
# A. a 3x4 subplot figure displaying a row per noise function and column per
#    experimental measurement (lens_diameter, lens_diameter_rel, lens_count, lens_count_rel)
measurements_exp = list(measurements)
measurements_exp = ['lens_diameter', 'lens_diameter_rel', 'lens_count', 'lens_count_rel']
# do this twice: with absolute and relative intensity measurements
for xval, title in zip(
        ['intensity', 'intensity_rel'],
        ['raw x-values', 'relative x-values']):
    colors = colors[:len(folders)]
    # fig, axes = plt.subplots(nrows=3, ncols=len(measurements_exp), figsize=(12, 9),
    #                          constrained_layout=True)
    fig, axes = plt.subplots(nrows=len(noise_functions), ncols=len(measurements_exp), figsize=(12, 6),
                             constrained_layout=True)
    fig.suptitle(f"ODA Results ({title.title()})")
    for row_num, (func, row) in enumerate(zip(noise_functions, axes)):  # row per function
        i = dataset.function.values == func
        sub_dataset = dataset.iloc[i]
        # color per folder
        for folder_num, (folder, color) in enumerate(zip(
                folders, colors)):  
            i = sub_dataset.folder.values == folder
            subsub_dataset = sub_dataset.iloc[i]
            # line per filename
            fnames = subsub_dataset.filename
            fnames = np.array([os.path.basename(os.path.dirname(os.path.dirname(fname))) for fname in fnames])
            fnames_set = sorted(set(fnames))
            for fname in fnames_set:
                i = fnames == fname
                subsubsub_dataset = subsub_dataset.iloc[i]
                for col_num, (measurement, ax) in enumerate(zip(measurements_exp, row)):
                    xvals = subsubsub_dataset[xval]
                    yvals = subsubsub_dataset[measurement]
                    ax.plot(xvals, yvals, color=color, alpha=.25)
                    ax.scatter(xvals, yvals, marker='.', color=color, edgecolors=None)
                    if folder_num == len(folders)-1:
                        no_nans = np.isnan(sub_dataset[measurement].values) == False
                        no_infs = np.isinf(sub_dataset[measurement].values) == False
                        include = no_nans * no_infs
                        ax.set_ylim(0, 1.1*np.nanmax(sub_dataset[measurement][include]))
                        # if 'rel' in measurement:
                        #     ax.semilogx()
                        ax.semilogx()
                        if measurement == 'lens_count_rel':
                            ax.set_ylim(0, 2)
                        elif measurement == 'lens_diameter_rel':
                            # ax.semilogy()
                            ax.set_ylim(0, 2)
                        for spine in ['top', 'right']:
                            ax.spines[spine].set_visible(False)
                        if func == 'reduce_size':
                            if xval == 'intensity':
                                ax.set_xlabel("Bin Size (pixels)")
                            else:
                                ax.set_xlabel("Pixel Resolution\n(pixels/ommatidium)")
                        elif func == 'gaussian_blur':
                            if xval == 'intensity':
                                ax.set_xlabel("Gaussian STD\n(pixels)")
                            else:
                                ax.set_xlabel("Scale Resolution\n(pixels/ommatidium)")
                        elif func == 'change_contrast':
                            ax.set_xlabel("Contrast (RMS)")
                        # ax.semilogy()
                        if col_num == 0:
                            ax.set_ylabel(func.replace("_", " ") + "\n" +
                                          measurement.replace("_", " "))
                        else:
                            ax.set_ylabel(measurement.replace("_", " "))
    # plt.tight_layout(rect=[0, 0.03, 1, 0.98], v_pad=3)
    # plt.savefig(f"results_{xval}.svg")
# plt.show()

# B. A 3x5 subplot figures displaying a row per noise function and column per duration
#    measurement per xvalue used (relative or absolute)
measurements_time = measurements[4:-1]
# do this twice: with absolute and relative intensity measurements
for xval, title in zip(
        ['intensity', 'intensity_rel'],
        ['raw x-values', 'relative x-values']):
    colors = colors[:len(folders)]
    fig, axes = plt.subplots(nrows=len(noise_functions), ncols=len(measurements_time), figsize=(12, 6),
                             constrained_layout=True)
    fig.suptitle(f"ODA Benchmarks ({title.title()})")
    for row_num, (func, row) in enumerate(zip(noise_functions, axes)):  # row per function
        i = dataset.function.values == func
        sub_dataset = dataset.iloc[i]
        # color per folder
        for folder_num, (folder, color) in enumerate(zip(
                folders, colors)):
            i = sub_dataset.folder.values == folder
            subsub_dataset = sub_dataset.iloc[i]
            # line per filename
            fnames = subsub_dataset.filename
            fnames = np.array([os.path.basename(os.path.dirname(os.path.dirname(fname))) for fname in fnames])
            fnames_set = sorted(set(fnames))
            for fname in fnames_set:
                i = fnames == fname
                subsubsub_dataset = subsub_dataset.iloc[i]
                for col_num, (measurement, ax) in enumerate(zip(measurements_time, row)):
                    xvals = subsubsub_dataset[xval]
                    yvals = subsubsub_dataset[measurement]
                    ax.plot(xvals, yvals, color=color, alpha=.25)
                    ax.scatter(xvals, yvals, marker='.', color=color, edgecolors=None)
                    if folder_num == len(folders)-1:
                        ax.set_ylim(0, 8)
                        ax.semilogx()
                        for spine in ['top', 'right']:
                            ax.spines[spine].set_visible(False)
                        if func == 'reduce_size':
                            if xval == 'intensity':
                                ax.set_xlabel("Bin Size (pixels)")
                            else:
                                ax.set_xlabel("Resolution\n(pixels/ommatidium)")
                        elif func == 'gaussian_blur':
                            if xval == 'intensity':
                                ax.set_xlabel("Gaussian STD\n(pixels)")
                            else:
                                ax.set_xlabel("Resolution\n(pixels/ommatidium)")
                        elif func == 'change_contrast':
                            ax.set_xlabel("Contrast")
                        # ax.semilogy()
                        if col_num == 0:
                            # ax.set_ylabel(func.replace("_", " ") + "\n" +
                            #               measurement.replace("_", " "))
                            ax.set_ylabel(measurement.replace("_", " "))
                        else:
                            ax.set_ylabel(measurement.replace("_", " "))
    # plt.tight_layout(rect=[0, 0.03, 1, 0.98], h_pad=3)
    # plt.savefig(f"benchmark_comparison_{title.replace(' ', '_')}.svg")
# plt.show()

# plot a summary figure with the relative lens count, the relative diameter measurement,
# and total duration (per row) for each result dataset

measurements_summary = ['lens_count_rel', 'lens_diameter_rel', 'total_dur']
measurements_titles = ['Relative\nLens Count', 'Relative\nLens Diameter', 'Total Duration (s)']
# do this twice: with absolute and relative intensity measurements
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(4, 6),
                         constrained_layout=True)
colors = colors[:len(folders)]
fig.suptitle(f"ODA Performance")
functions = [noise_functions[0], noise_functions[1]]
for col_num, (func, col) in enumerate(zip(functions, axes.T)):  # row per function
    i = dataset.function.values == func
    sub_dataset = dataset.iloc[i]
    # color per folder
    for folder_num, (folder, color) in enumerate(zip(
            folders, colors)):
        i = sub_dataset.folder.values == folder
        subsub_dataset = sub_dataset.iloc[i]
        # line per filename
        fnames = subsub_dataset.filename
        fnames = np.array([os.path.basename(os.path.dirname(os.path.dirname(fname))) for fname in fnames])
        fnames_set = sorted(set(fnames))
        for fname in fnames_set:
            i = fnames == fname
            subsubsub_dataset = subsub_dataset.iloc[i]
            for row_num, (measurement, ax, title) in enumerate(zip(
                    measurements_summary, col, measurements_titles)):
                xvals = subsubsub_dataset[xval]
                yvals = subsubsub_dataset[measurement]
                ax.plot(xvals, yvals, color=color, alpha=.25, zorder=1)
                ax.scatter(xvals, yvals, marker='.', color=color,
                           edgecolors=None, zorder=1)
                if folder_num == len(folders)-1:
                    ax.semilogx()
                    for spine in ['top', 'right']:
                        ax.spines[spine].set_visible(False)
                    for spine in ['left', 'bottom']:
                        ax.spines[spine].set_zorder(3)
                    if row_num == 2:
                        ax.set_ylim(0, 8)
                        ax.set_yticks([0, 4, 8])
                        if col_num == 0:
                            ax.set_xlabel("Resolution\n(pixels/ommatidium)")
                        else:
                            ax.set_yticklabels([])
                            ax.set_xlabel("Contrast (RMS)")
                    else:
                        ax.set_ylim(0, 2)
                        ax.set_yticks([0, 1, 2])
                        ax.set_xticklabels([])
                    if col_num == 0:
                        ax.set_xlim(.25, 100)
                        ax.set_ylabel(title)
                        ax.axvspan(0, 2, color='.8', zorder=0)
                        # ax.axvline(2, color='k', linestyle='--', zorder=1, lw=1)
                    # elif row_num != 2:
                    elif row_num != len(noise_functions):
                        # ax.set_yticklabels([])
                        ax.set_xticklabels([])
                        ax.spines['left'].set_visible(False)
                        ax.set_yticks([])
                    else:
                        ax.spines['left'].set_visible(False)
                        ax.set_yticks([])
plt.savefig(f"benchmark_summary.svg")
plt.show()
# logistically regress ommatidial count on relative intensity measurements
# with folder and noise function as covariates. For example, for each function:
# lens_diameter_rel ~ intensity_rel * folder

# go through the dataset and model the relative diameters and counts as
# a logistic function of relative intensity
# 1. add a subject column
subjects = []
for num, row in dataset.iterrows():
    fn = row.filename
    subject = os.path.dirname(os.path.dirname(fn))
    subjects += [subject]
dataset['subject'] = pd.Series(subjects, dtype='str')
subject_set = np.unique(subjects)
# 2. for each subject, approximate a logistic function by a) subtracting the minimum,
#    b) dividing by the resulting maximum, and c) reporting the first x-value when the
#    y-value exceeds .5 (the tipping point)
resolution_results = {
    'subject':[], 'function':[], 'folder':[], 
    'count_min':[], 'count_max':[], 'count_thresh':[], 'count_best':[],
    'diam_min':[], 'diam_max':[], 'diam_thresh':[], 'diam_best':[],
    'count_best':[], 'diam_best':[]}
contrast_results = {
    'subject':[], 'function':[], 'folder':[], 
    'count_min':[], 'count_max':[], 'count_thresh':[], 'count_best':[],
    'diam_min':[], 'diam_max':[], 'diam_thresh':[], 'diam_best':[]}

for noise_func, results in zip(['reduce_size', 'change_contrast'],
                               [resolution_results, contrast_results]):
    # fig = plt.figure()
    # get inds in dataset with same function
    func_inds = dataset['function'].values == noise_func
    # for each subject, 
    for subject in subject_set:
        subject_inds = dataset['subject'].values == subject
        inds = func_inds * subject_inds
        # get data for the subject and noise function
        subject_data = dataset[inds]
        # store subject related data
        results['subject'] += [subject]
        results['function'] += [noise_func]
        results['folder'] += [subject_data.folder.values[0]]
        # get xvals, ommatidial diameter, and count data
        xvals = subject_data.intensity_rel.values
        diameters = np.copy(subject_data.lens_diameter_rel.values)
        counts = np.copy(subject_data.lens_count_rel.values)
        # store minima, crossover point, and maxima
        for vals, var in zip([diameters, counts],
                             ['diam', 'count']):
            # the best estimates are the last in the array
            best_ind = np.argmax(xvals)
            results[f"{var}_best"] += [np.copy(vals[best_ind])]
            minimum = np.nanmin(vals)
            maximum = np.nanmax(vals[vals < np.inf])
            # to get the crossover, normalize the vals
            vals -= minimum
            vals /= np.nanmax(vals)
            # test: plot the curves 
            # plt.plot(xvals, vals, alpha=.2)
            # plt.semilogx()
            # find xvals where vals > .5
            if var == 'diam':
                low_vals = vals < .5
                if any(low_vals):
                    thresh = xvals[low_vals].min()
                else:
                    breakpoint()
            if var == 'count':
                high_vals = vals > .5
                if any(high_vals):
                    thresh = xvals[high_vals].min()
                else:
                    breakpoint()
            # store data
            results[f"{var}_min"] += [minimum]
            results[f"{var}_max"] += [maximum]
            results[f"{var}_thresh"] += [thresh]
contrast_results = pd.DataFrame(contrast_results)
contrast_results.to_csv("contrast_results.csv")
resolution_results = pd.DataFrame(resolution_results)
resolution_results.to_csv("resolution_results.csv")

# todo: box plot the contrast and resolution results in two subplots
# top: (left col) minimum and (middle) maximum contrast response and (right) threshold contrast
# bottom: (left col) minimum and (middle) maximum resolution response and (right) threshold contrast


# 1. make a figure with 2 rows and 3 columns
# graphing the following:
# fig, axes = plt.subplots(ncols=2, nrows=2)
# a. dataset (resolution vs. contrast)         --> top vs. bottom subplot rows
jitter = .05
# for results, row in zip([contrast_results, resolution_results],
#                         axes):
for results, title in zip(
        [contrast_results, resolution_results],
        ['Contrast', 'Resolution']):
    # b. variable (count vs. diameter)         --> left vs. right subplot columns
    fig, row = plt.subplots(ncols=6)
    fig.suptitle(title)
    # for ax, var in zip(row, ['diam', 'count']):
    for axes, var in zip([row[:3], row[3:]], ['diam', 'count']):
        # c. measurement (min, max, thresh)    --> x-position in subplot
        # get xpositions so that the boxplots are grouped by measurement
        xvals = np.array([0, 1, 2])  # 3 measurements: min, max, and thresh
        # d. folder                            --> color
        xdiffs = .1 * np.arange(len(set(results.folder))).astype(float)
        xdiffs -= xdiffs.mean()
        xvals = xvals[:, np.newaxis] + xdiffs
        for measurement, xs, ax in zip(
                ['min', 'max', 'thresh'],
                xvals, axes):
            full_var = f"{var}_{measurement}"
            # grab the results to plot
            yvals = results[full_var]
            groups = results.folder
            # for each group and xval, plot a boxplot
            for x, folder, color in zip(xs, folders, colors):
                include = groups == folder
                ys = yvals[include]
                # plot the mean
                ax.plot(x, ys.mean(), marker='o', color=color, label=folder)
                ax.set_title(measurement.title())
                # plot the points with some xjitter
                x_jitter = np.repeat(x, len(ys))
                x_jitter += jitter * (np.random.random(len(ys)) - .5)
                ax.scatter(x_jitter, ys, color=color, alpha=.5, edgecolors='none', marker='.')
            ax.legend()
# plt.show()

# 2. make 2 scatterplots: 1) reference count by output measured count, 2
# get the maximum intensity dataset
max_inds = []
subjects = np.array(subjects)
noise_types = np.array(dataset.function.values)
intensity_vals = np.array(dataset.intensity.values)
for subject in subject_set:
    subject_inds = subjects == subject
    func_inds = noise_types == 'reduce_size'
    intensity_inds = intensity_vals == 1
    inds = subject_inds * func_inds * intensity_inds
    ind = np.where(inds)[0][0]
    max_inds += [ind]
max_inds = np.array(sorted(max_inds))
dataset_max = dataset.iloc[max_inds]
# scatterplot of automatic and measured counts and diameters, color coded by the folder
fig, axes = plt.subplots(ncols=2)
ycols = ['reference_diam', 'reference_count']
xcols = ['lens_diameter', 'lens_count']
colorvals = dataset.folder.values
colorvals_set, colorvals_lbls = np.unique(colorvals, return_inverse=True)
for ax, ycol, xcol in zip(axes, ycols, xcols):
    # make an ols model with folder as a dummy-coded variable
    # print(f"{ycol} ~ {xcol} + folder")
    # ols = smf.ols(f"{ycol}~{xcol}+folder", data=dataset_max).fit()
    # ols_simple = smf.ols(f"{ycol}~{xcol}", data=dataset_max).fit()
    # print(ols_simple.summary())
    print(f"corr({ycol} ~ {xcol})")
    for folder, color in zip(folders, colors):
        # get subset of points from individual folders
        include = dataset_max.folder == folder
        sub_dataset_max = dataset_max[include]
        xvals = sub_dataset_max[xcol]
        yvals = sub_dataset_max[ycol]
        ax.scatter(xvals, yvals, color=color, zorder=3, alpha=.5, edgecolors='none')
        # plot the modeled values per subset
        # xs = np.array([xvals.min(), xvals.max()])
        # new_yvals = ols.predict({xcol: xs, 'folder': np.repeat(folder, 2)})
        # ax.plot(xs, new_yvals, color=color, zorder=2)
        # print the correlation for this subset
        corr, p = stats.pearsonr(xvals, yvals)
        print(f"{folder}: {np.round(corr, 3)} (P={np.round(p, 3)})")
    maxy = max(dataset_max[xcol].max(), dataset_max[ycol].max())
    miny = min(dataset_max[xcol].min(), dataset_max[ycol].min())
    ax.plot([miny, maxy], [miny, maxy], color='k', zorder=1)
    ax.set_aspect('equal')
plt.tight_layout()
plt.show()
