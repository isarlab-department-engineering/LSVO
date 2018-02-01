import os
import time

import matplotlib.pyplot as plt
import numpy as np
from objectives.VoObjectives import rmse_error

from utils.rel2abs import rel2abs


class AbstractModel(object):
    def __init__(self, config):
        self.config = config
        self.dataset = {}
        self.training_set = {}
        self.test_set = {}
        self.model = self.build_model()
        self.prepare_data()

    def prepare_data(self):

        raise ("Not implemented - this is an abstract method")

    def test_data_generator(self):

        curr_test_sample = 0

        while 1:
            if (curr_test_sample == len(self.test_set)):
                break

            x_test = self.test_set[curr_test_sample].read_features()
            y_test = self.test_set[curr_test_sample].read_labels()
            y_test = np.expand_dims(y_test, axis=0)
            x_test, y_test = self.prepare_data_for_model(x_test, y_test)

            curr_test_sample += 1

            yield x_test, y_test

    def prepare_data_for_model(self, features, label):

        raise ("Not implemented - this is an abstract method")

    def specific_test_processing(self, pred, true, results, gt):

        raise ("Not implemented - this is an abstract method")

    def build_model(self):

        raise ("Not implemented - this is an abstract method")

    def save_prediction(self, T, filename, architecture):

        result_dir = 'results'
        n = T.shape[0]
        t = np.ndarray((n, 12))

        for k in range(n):
            t[k, :] = np.ravel(T[k, 0:3, :])
        arch_traj_dir = os.path.join(result_dir, 'trajectories', architecture)
        if not os.path.exists(arch_traj_dir):
            os.makedirs(arch_traj_dir)
        print(arch_traj_dir)
        np.savetxt(os.path.join(arch_traj_dir, filename + '.txt', ), t, fmt='%.6e')

    def test(self, weights_file):

        print("Testing model")
        self.model.load_weights(weights_file)
        test_seqs = []

        if self.config.use_Kitti:
            print('--------------------------------------')
            print('------Testing on Kitti Dataset--------')
            print('--------------------------------------')
            test_seqs = self.dataset['kitti'].generate_test_data_by_sequence()


        if self.config.use_Malaga:
            print('--------------------------------------')
            print('------Testing on malaga Dataset--------')
            print('--------------------------------------')
            test_seqs = self.dataset['malaga'].generate_test_data_by_sequence()
            count_seq_for_kitti = 15

        for seq in test_seqs:
            print('--------------------------------------')
            print('------Testing Sequence: {}'.format(test_seqs[seq].sequence_name))

            sum_error = 0
            sum_time = 0

            curr_img = 0
            results = []
            gt = []

            for sample in test_seqs[seq].generated_sample:
                print('------Testing Sample: {}'.format(curr_img))
                x_test = sample.read_features()
                y_test = sample.read_labels()
                y_test = np.expand_dims(y_test, axis=0)
                x_test, y_test = self.prepare_data_for_model(x_test, y_test)

                x_test = np.expand_dims(x_test, axis=0)
                t0 = time.time()
                output = self.model.predict(x_test)
                t1 = time.time()
                sum_time += (t1-t0)

                gt, results = self.specific_test_processing(output, y_test, results, gt)
                curr_img += 1

                sum_error += rmse_error(gt[-1, :], results[-1, :])

            print('Avg rmse: {}'.format(sum_error/len(test_seqs[seq].generated_sample)))
            print('Avg time: {}'.format(sum_time / len(test_seqs[seq].generated_sample)))

            Tl, T = rel2abs(results)
            curr_seq_number = test_seqs[seq].sequence_name.split('/')[1]

            if self.config.use_Kitti:
                curr_seq_number = curr_seq_number.split('_')[0]
            elif self.config.use_Malaga:
                curr_seq_number = str(count_seq_for_kitti)
                count_seq_for_kitti += 1

            self.save_prediction(T, curr_seq_number, self.config.strategy)

            trj00 = np.mat(T[:, 0:3, 3])
            gtTl, gtT = rel2abs(gt)
            trj00gt = np.mat(gtT[:, 0:3, 3])

            plt.figure()
            plt.plot(trj00gt[:, 0], trj00gt[:, 2], color='#FF00FF', label='GT', lw=2)
            plt.plot(trj00[:, 0], trj00[:, 2], color='#0000FF', label="CNNVO-1b", lw=2)
            plt.show()

