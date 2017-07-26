import os
import numpy as np
from scipy import ndimage
import pickle
import pathlib


def normalize(image_data, pixel_depth=255.0):
    return (image_data - pixel_depth/2) / pixel_depth


def validate(image_data, image_size=28):
    if image_data.shape != (image_size, image_size):
        error_msg = ('Unexpected image shape: {}'.format(image_data.shape))
        raise Exception(error_msg)
    return image_data


def load_images(image_files):
    for image_file in image_files:
        try:
            yield validate(normalize(ndimage.imread(image_file)))
        except IOError as e:
            print("Could not read:", image_file, "- it's ok, skipping.")
            continue


def load_letter(src_dir):
    image_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    return np.stack(load_images(image_files))


def piclke_letter(src_dir, dest_file):
    with open(dest_file, 'wb') as f:
        dataset = load_letter(src_dir)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def pickle_letters(src_dir=None, dest_dir=None):
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    for letter in os.listdir(src_dir):
        letter_dir = os.path.join(src_dir, letter)
        letter_pickle_file = os.path.join(dest_dir, letter) + '.pickle'

        if os.path.isfile(letter_pickle_file):
            print("File already exists: {}".format(letter_pickle_file))
        else:
            piclke_letter(letter_dir, letter_pickle_file)
