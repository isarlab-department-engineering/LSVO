import os
from glob import glob
import numpy as np
from itertools import tee, islice


class Sequence(object):

    def __init__(self, directory, extension, label_file, dir, is_grayscale = False, name='Sequence', of_dir='optical_flow', of_ext='flo'):

        self.sequence_name = name
        self.sequence_dir = directory
        self.is_grayscale = is_grayscale
        self.image_paths = []
        self.dir = dir
        self.of_paths = []
        self.label = []
        self.generated_sample = []

        self.load_img_paths(extension)
        self.load_of_paths(of_dir, of_ext)
        self.load_label(label_file)

    def load_img_paths(self, extension):
        self.image_paths = sorted(glob(os.path.join(self.sequence_dir, '*' + '.' + extension)))

    def load_of_paths(self, of_dir, of_ext):
        self.of_paths = sorted(glob(os.path.join(self.sequence_dir, of_dir, '*' + '.' + of_ext)))

    def load_label(self, label_file):

        if os.path.exists(label_file):
            self.label = np.loadtxt(os.path.join(self.sequence_dir, label_file))
        else:
            self.label = np.zeros((len(self.image_paths), 6))

    def get_num_imgs(self):
        return len(self.image_paths)

    def get_num_label(self):
        return len(self.label)

    def get_image_paths(self):
        return self.image_paths

    def get_dir(self):
        return self.dir

    def get_of_paths(self):
        if len(self.of_paths) == 0:
            raise Exception("No optical flow found for sequence: ", self.sequence_dir)
        return self.of_paths

    def get_labels(self):
        return self.label

    def get_is_grayscale(self):
        return self.is_grayscale

    def set_generated_sample(self, generated_sample):
        self.generated_sample = generated_sample

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return zip(*iters)
