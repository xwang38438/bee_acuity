This folder contains the input and output of the benchmark analysis used in https://doi.org/10.1101/2020.12.11.422154.

The following folders containing the original images used in the ODA 2D benchmark, grouped by their dataset:
  * ./ant_replicas                  <!-- 5 micrographs of different ant species -->
  * ./D_melanogaster_micrographs    <!-- 5 micrographs of different D. melanogaster individuals -->
  * ./D_melanogaster_SEM            <!-- 5 SEMs of different D. melanogaster individuals -->
  * ./D_mauritiana_SEM				<!-- 5 SEMs of different D. mauritiana individuals -->

To generate the images for benchmarking the effect of contrast and image resolution, run benchmark.py. The program runs on Python 3 and depends on the following open source modules that can all be installed using pip:
  * ODA
  * matplotlib
  * numpy
  * pandas
  * scipy

This generates degraded versions of each image in each folder, reducing spatial resolution by averaging the image into 1, 2, 4, 8, 16, and 32 pixel square bins and contrast by reducing the range of pixel values from 1, .5, .25, .125, .062, and .031 of the original range around the mean value. Since the pixel values are discrete 8-bit integers, reducing the range reduces the root mean square of contrasts in the image. The program also performs a benchmark test excluded from the paper, which reduces resolution by applying a gaussian blur instead of bin-averaging. After generating each image, the cProfile module is used to benchmark the performance of the ODA on each degraded image.

Within each folder, there is a subfolder containing the mask images indicating the pixels in the image corresponding to the eye and files used for converting distances from pixel lengths to micrometers.

Running benchmark.py produces the following:

For each image in each folder, a new subfolder is created named after the corresponding file. Within each of these image folders are 3 subfolders, one for each degradation experiment, and 3 images summarizing the results of those experiments. Within each of the experiment subfolders (./change_contrast, ./reduce_size, and ./gaussian_blur) are:
  * degraded images with filenames indicating the filter intensity, such as "intensity_1.000.png"; 
  * .txt files with the same basename indicating the resolution or contrast (rms) of the corresponding image
  * 4 folders containing important stages of the ODA applied to the degraded images: 
	* ./reciprocal -- the reciprocral image used for determing the fundamental frequency
	* ./filtered   -- the low-pass filtered image used to find ommatidia centers
	* ./ommatidia  -- the test image with the output ommatidial centers plotted
	* ./stats      -- the profiler stats on the particular folder
* 1 folder containing the profiler statistics of running the ODA on each degraded image
  * ./data.csv     -- a spreadsheet containing the benchmark results for each degraded image
  
For more information, see the docstring at the top of benchmark.py.

To accumulate and plot the results of the benchmark, run analysis.py. Among others, it will plot the benchmark results summary presented in the paper. This will also create two spreadsheets corresponding to the two reported degradation experiments:
  * contrast_results.csv   -- one row for each image with columns indicating the minimum and maximum counts and diameters expressed as proportions of the manual measurements over the range of contrasts tested. It also includes the threshold contrast for counts and diameters representing when the measure crosses half of the maximum.
  * resolution_results.csv -- same as above but for changes in resolution.

Finally, all of the results for the benchmark figure are found in benchmark_summary.svg.
