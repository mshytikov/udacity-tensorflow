import os
import numpy as np
from scipy import ndimage
import pickle
import pathlib
import collections

from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle

RANDOM_STATE = np.random.RandomState(10)

Dataset = collections.namedtuple('Dataset', ['x', 'y'])
Workset = collections.namedtuple('Workset', ['train', 'valid', 'test'])


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


def unpickle_letters(src_dir):
    for file_name in sorted(os.listdir(src_dir)):
#        letter = os.path.splitext(file_name)[0]
        with open(os.path.join(src_dir, file_name), 'rb') as f:
            yield pickle.load(f)


def split_dataset(X, train_size=None, test_size=None):
    shuffle_split = ShuffleSplit(
            n_splits=1,
            random_state=RANDOM_STATE,
            train_size=train_size,
            test_size=test_size,
            )
    for train_index, test_index in shuffle_split.split(X):
        return [X[train_index], X[test_index]]


def build_datasets(src_dir, train_size, test_size):
    letters = [l for l in unpickle_letters(src_dir)]
    train_size_per_leter = train_size // len(letters) + 1
    test_size_per_leter = test_size // len(letters) + 1

    train_datasets = []
    train_labels = []
    test_datasets = []
    test_labels = []

    for index, letter in enumerate(letters):
        train, test = split_dataset(
                letter,
                train_size=train_size_per_leter,
                test_size=test_size_per_leter
                )

        train_datasets.append(train)
        train_labels.append(np.full(train_size_per_leter, index))
        test_datasets.append(test)
        test_labels.append(np.full(test_size_per_leter, index))

    train_x = np.concatenate(train_datasets)[:train_size]
    train_y = np.concatenate(train_labels)[:train_size]
    test_x = np.concatenate(test_datasets)[:test_size]
    test_y = np.concatenate(test_labels)[:test_size]

    train = Dataset(*shuffle(train_x, train_y))
    test = Dataset(*shuffle(test_x, test_y))

    return (train, test)


def pickle_workset(dest_file, workset):
    with open(dest_file, 'wb') as f:
        pickle.dump(workset, f, pickle.HIGHEST_PROTOCOL)


def unpickle_workset(src_file):
    with open(src_file, 'rb') as f:
        return pickle.load(f)
