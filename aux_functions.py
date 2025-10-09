import numpy as np
from skimage import exposure


def convert_to_dB(array: np.array):
    """Convert np array to dB.
    Assumes amplitude values and clips values to
    the minimum positive value present in the data."""
    # find a positive min number to replace zeros
    abs_min = np.abs(np.min(array[np.where(array > 0)]))
    array[np.where(array == 0)] = abs_min
    # clip values to that minimum
    if np.nanmin(array) <= 0:
        array = np.clip(array, a_min=abs_min, a_max=None)
    array = 20 * np.log10(array)
    return array


def hist_equalization(array: np.array):
    """Wrapper function to apply CLAHE with clip_limit=0.01"""
    # if negative values are present, shift all values up
    if np.nanmin(array) < 0:
        array -= np.nanmin(array)
    # Prevent 0 in array
    if np.nanmin(array) == 0:
        array += np.nanmin(array[np.where(array != 0)])
    array /= np.nanmax(np.abs(array))
    array = exposure.equalize_adapthist(array, clip_limit=0.01)
    return array
