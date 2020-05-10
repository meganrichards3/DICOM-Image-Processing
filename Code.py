'''
Code Adapted From:
Ski-Cit Image Tutorials (https://scikit-image.org/docs/dev/auto_examples/index.html)
Ski-Py Lectures (https://scipy-lectures.org/index.html)
Medium Tutorial/Guide to Image Processing: https://medium.com/@hengloose/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed
Radiology Data Tutorial: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python'''

import numpy as np
import pydicom
import os
from glob import glob
from scipy.ndimage import gaussian_filter
from skimage.filters import unsharp_mask
from skimage.morphology import reconstruction
from plotly.offline import init_notebook_mode
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import filters

#loads DICOM scan from datapath
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

#takes in full scan (all slices) and converts to HU units, returns np array of the scan pixel arrays
#first index=slice number, second index=indexing slice pixels
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


#prints a sample stack of slices. Takes in the whole set, indexes with show_every, and displays the amount that fits in rows*cols
def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=8):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')
    plt.show()


#take in set and index and displays that slice
def sample_slice(imgs_to_process, ind):
    plt.imshow(imgs_to_process[ind], cmap='gray')
    plt.show()


# takes in set, plots a histogram of the HU values for the whole set by frequency
def plot_hu_histogram(imgs_to_process):
    plt.hist(imgs_to_process.flatten(), bins=50, color='c', label="Hounsfield Units for Scan")
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


# takes in two image arrays and presents them side by side
def print_comparison(original, new):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(new, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# takes in the image set and the index and uses a gaussian filter to blur the image and print a comparison
def blur(imgs_to_process, index):
    blurred_f = ndimage.gaussian_filter(imgs_to_process[index], 3)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
    alpha = 30
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
    print_comparison(blurred_f, sharpened)


# applies roberts filter to all images
def roberts_filter(imgs_to_process):
    for i in range(len(g)):
        imgs_to_process[i] = filters.roberts(imgs_to_process[i])
    print_comparison(imgs_to_process[10], imgs_to_process[100])

# takes in the image set, index of slice, threshold value, and alpha multiplier. Checks indexed array for values below
# threshold and multiplies it by alpha, returning the adapted slice.
def darken_implant(imgs, index, threshold, alpha):
    for i in range(len(imgs[index])):
        for j in range(len(imgs[index][i])):
            if imgs[index][i][j] < threshold:
                imgs[index][i][j] = imgs[index][i][j]*alpha
    return imgs[100]

#Takes in image set and index, applies a gaussian filter and mask to identify and remove excess elements, printing the
#results
def remove_excess (imgs_to_process, index):
    image = gaussian_filter(imgs_to_process[index], 1)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5), sharex=True, sharey=True)

    ax0.imshow(image, cmap='gray')
    ax0.set_title('original image')
    ax0.axis('off')

    ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
    ax1.set_title('dilated')
    ax1.axis('off')

    ax2.imshow(image - dilated, cmap='gray')
    ax2.set_title('image - dilated')
    ax2.axis('off')

    fig.tight_layout()
    plt.show()


# takes in image set and index of slice, performing a sharpening mask and printing the adapted image.
def sharpen (imgs_to_process, index):
    image = imgs_to_process[index]
    result_1 = unsharp_mask(image, radius=1, amount=1)
    result_2 = unsharp_mask(image, radius=5, amount=2)
    result_3 = unsharp_mask(image, radius=20, amount=1)

    fig, axes = plt.subplots(nrows=2, ncols=2,
                             sharex=True, sharey=True, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    ax[1].imshow(result_1, cmap=plt.cm.gray)
    ax[1].set_title('Enhanced image, radius=1, amount=1.0')
    ax[2].imshow(result_2, cmap=plt.cm.gray)
    ax[2].set_title('Enhanced image, radius=5, amount=2.0')
    ax[3].imshow(result_3, cmap=plt.cm.gray)
    ax[3].set_title('Enhanced image, radius=20, amount=1.0')

    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()

#Some Example Code

data_path = r"C:\Users\Megan\Desktop\Ind Study\Set 2"  # make sure the folder with the whole set is opened, and that the r in the front isn't deleted
output_path = working_path = r"C:\Users\Megan\Desktop\Ind Study\results"
g = glob(data_path + '/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print('\n'.join(g[:5]))

# load patient data, converts to HU units, saves image as .npy
ID = 0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)
np.save(output_path + "fullimages_%d.npy" % ID, imgs)

# gets image from stored path, puts it in imgs_to_process
file_used = output_path + "fullimages_%d.npy" % ID
imgs_to_process = np.load(file_used).astype(np.float64)

#a few function call examples
#remove_excess (imgs_to_process, 100)
#print_comparison(imgs_to_process[100], imgs_to_process[101])
#plot_hu_histogram(imgs_to_process)




