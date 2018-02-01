import os
from dataset.Sequence import Sequence
from collections import OrderedDict
import numpy as np


class Dataset(object):

    def __init__(self, config, data_generation_strategy, name):
        self.config = config
        self.name = name
        self.data_generation_strategy = data_generation_strategy
        self.training_seqs = OrderedDict()
        self.test_seqs = OrderedDict()

    def read_data(self):

        raise Exception("Not implemented - this is an abstract method")

    def extract_of(self):

        for sequence in self.training_seqs:
            self.training_seqs[sequence].compute_optical_flow(self.config)
        for sequence in self.test_seqs:
            self.test_seqs[sequence].compute_optical_flow(self.config)

    def get_of_by_sequence(self, sequence):

        return self.training_seqs[sequence].get_ofs(self.config)



    def print_info(self):

        print('--------------------------------------')
        print('------Dataset Info--------')
        print('Dataset Name: {}'.format(self.name))
        print('Number of Training dirs: {}'.format(len(self.training_seqs)))
        print('Training dirs:')
        for directory in self.training_seqs:
            curr_sequence = self.training_seqs[directory]
            print(directory,
                  curr_sequence.sequence_dir,
                  'Num imgs: {}'.format(curr_sequence.get_num_imgs()),
                  'Num label: {}'.format(curr_sequence.get_num_label()))

        print('Number of Test dirs: {}'.format(len(self.test_seqs)))
        print('Test dirs:')
        for directory in self.test_seqs:
            curr_sequence = self.test_seqs[directory]
            print(directory,
                  curr_sequence.sequence_dir,
                  'Num imgs: {}'.format(curr_sequence.get_num_imgs()),
                  'Num label: {}'.format(curr_sequence.get_num_label()))


    def get_seq_stat(self):

        raise Exception("Not implemented - this is an abstract method")

    def generate_train_test_data(self):

        train_data = self.data_generation_strategy.generate_data(self.training_seqs)
        test_data = self.data_generation_strategy.generate_data(self.test_seqs)
        return train_data, test_data

    def generate_test_data_by_sequence(self):
        for sequence in self.test_seqs:
            curr_generated_samples = self.data_generation_strategy.get_image_set(self.test_seqs[sequence])
            self.test_seqs[sequence].set_generated_sample(curr_generated_samples)
        return self.test_seqs


class KittiDataset(Dataset):

    def __init__(self, config, data_generation_strategy=None):
        super(KittiDataset, self).__init__(config, data_generation_strategy, 'Kitti')

    def read_data(self):
        if self.config.use_subsampled:
            subdir = 'image_0/downsampled_' + str(self.config.input_height) + '_' + str(self.config.input_width)
        else:
            subdir = 'image_0'


        for dir in self.config.kitti_train_dirs:
            seq_dir = os.path.join(self.config.data_set_dir, self.config.kitti_main_dir, dir)
            self.training_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                               extension='png',
                                               label_file=os.path.join(seq_dir, 'label' + dir  + '.txt'),
                                               dir=dir,
                                               is_grayscale=self.config.kitti_is_grayscale,
                                               name='Kitti_train/' + dir,
                                               of_dir=self.config.kitti_of_dir,
                                               of_ext=self.config.kitti_of_ext)

        for dir in self.config.kitti_test_dirs:
            seq_dir = os.path.join(self.config.data_set_dir, self.config.kitti_main_dir, dir)
            self.test_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                           extension='png',
                                           label_file=os.path.join(seq_dir, 'label' + dir  +'.txt'),
                                           dir=dir,
                                           is_grayscale=self.config.kitti_is_grayscale,
                                           name='Kitti_test/' + dir,
                                           of_dir=self.config.kitti_of_dir,
                                           of_ext=self.config.kitti_of_ext)

    def get_seq_stat(self):
        num_tr_seq = len(self.training_seqs)
        num_te_seq = len(self.test_seqs)
        num_tr_imgs = 0
        num_te_imgs = 0
        for seq in self.training_seqs:
            num_tr_imgs += self.training_seqs[seq].get_num_imgs()
        for seq in self.test_seqs:
            num_te_imgs += self.test_seqs[seq].get_num_imgs()
        return num_tr_seq, num_te_seq, num_tr_imgs, num_te_imgs

class MalagaDataset(Dataset):

    def __init__(self, config, data_generation_strategy=None):
        super(MalagaDataset, self).__init__(config, data_generation_strategy, 'Malaga')

    def read_data(self):
        if self.config.use_subsampled:
            subdir = 'image_0/downsampled_' + str(self.config.input_height) + '_' + str(self.config.input_width)
        else:
            subdir = 'image_0'


        for dir in self.config.malaga_train_dirs:
            seq_dir = os.path.join(self.config.data_set_dir, self.config.malaga_main_dir, dir)
            self.training_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                               extension='jpg',
                                               label_file=os.path.join(seq_dir, 'labels' + dir  + '.txt'),
                                               dir=dir,
                                               is_grayscale=self.config.malaga_is_grayscale,
                                               name='Malaga_train/' + dir,
                                               of_dir=self.config.malaga_of_dir,
                                               of_ext=self.config.malaga_of_ext)

        for dir in self.config.malaga_test_dirs:
            seq_dir = os.path.join(self.config.data_set_dir, self.config.malaga_main_dir, dir)
            self.test_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                           extension='jpg',
                                           label_file=os.path.join(seq_dir, 'labels' + dir  +'.txt'),
                                           dir=dir,
                                           is_grayscale=self.config.malaga_is_grayscale,
                                           name='Malaga_test/' + dir,
                                           of_dir=self.config.malaga_of_dir,
                                           of_ext=self.config.malaga_of_ext)

    def get_seq_stat(self):
        num_tr_seq = len(self.training_seqs)
        num_te_seq = len(self.test_seqs)
        num_tr_imgs = 0
        num_te_imgs = 0
        for seq in self.training_seqs:
            num_tr_imgs += self.training_seqs[seq].get_num_imgs()
        for seq in self.test_seqs:
            num_te_imgs += self.test_seqs[seq].get_num_imgs()
        return num_tr_seq, num_te_seq, num_tr_imgs, num_te_imgs
