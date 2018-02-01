from dataset.SampleType import OpticalFlowSample
import numpy as np
from itertools import tee, islice


class SampleGenerationStrategy(object):

    def __init__(self, ram_pre_loading):
        self.ram_pre_loading = ram_pre_loading

    def generate_data(self, sequences):

        img_sets = []
        for seq in sequences:
            print(seq)
            curr_seq = sequences[seq]
            seq_set = self.get_image_set(curr_seq)
            img_sets = np.append(img_sets, seq_set)

        return img_sets

    def get_image_set(self, sequence):

        raise ("Not implemented - this is an abstract method")

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return zip(*iters)


class OpticalFlowGenerationStrategy(SampleGenerationStrategy):
    def __init__(self, config, ram_pre_loading=False):

        self.config = config

        super(OpticalFlowGenerationStrategy, self).__init__(ram_pre_loading)

    def get_image_set(self, sequence):
        pairs = []

        for of, label_set in zip(sequence.get_of_paths(), sequence.get_labels()):
            pairs.append(
                OpticalFlowSample(self.config.input_height, self.config.input_width, of, label_set, self.ram_pre_loading))
        return pairs


