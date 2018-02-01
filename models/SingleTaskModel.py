
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import numpy as np

from dataset.Dataset import KittiDataset, MalagaDataset
from objectives.VoObjectives import rmse, vo_loss
from models.AbstractModel import AbstractModel
from dataset.DataGenerationStrategy import OpticalFlowGenerationStrategy


class SingleTaskModel(AbstractModel):

    def __init__(self, config):
        super(SingleTaskModel, self).__init__(config)

    def prepare_data(self):
        print('--------------------------------------')
        print('-------------SingleTaskModel---------------')
        print('--------------------------------------')
        if self.config.use_Kitti:
            print('--------------------------------------')
            print('------Processing Kitti Dataset--------')
            print('--------------------------------------')
            self.dataset['kitti'] = KittiDataset(self.config, OpticalFlowGenerationStrategy(self.config, ram_pre_loading=self.config.ram_pre_loading))
            self.dataset['kitti'].read_data()
            self.dataset['kitti'].print_info()
            self.training_set, self.test_set = self.dataset['kitti'].generate_train_test_data()

        if self.config.use_Malaga:
            print('--------------------------------------')
            print('------Processing Malaga Dataset--------')
            print('--------------------------------------')
            self.dataset['malaga'] = MalagaDataset(self.config, OpticalFlowGenerationStrategy(self.config, ram_pre_loading=self.config.ram_pre_loading))
            self.dataset['malaga'].read_data()
            self.dataset['malaga'].print_info()
            self.training_set, self.test_set = self.dataset['malaga'].generate_train_test_data()

    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')
        features /= 255
        label = np.asarray(label)
        return features, label

    def specific_test_processing(self, pred, true, results, gt):
        output = pred

        if len(results) == 0:
            results = output
        else:
            results = np.concatenate((results, output))

        output_gt = true

        if len(gt) == 0:
            gt = output_gt
        else:
            gt = np.concatenate((gt, output_gt))

        return gt, results


    def build_model(self):

        input_shape = Input(shape=(self.config.input_height,
                                   self.config.input_width,
                                   self.config.input_channel*2))

        conv1 = Convolution2D(64, kernel_size=(3, 3), padding='valid',
                              kernel_initializer='glorot_normal', strides=(2, 2), name='conv1')(input_shape)
        pool1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool1')(conv1)
        conv2 = Convolution2D(20, kernel_size=(3, 3), padding='valid', kernel_initializer='glorot_normal',
                              name='conv2')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(conv2)
        flatten1 = Flatten(name='flatten1')(pool1)
        flatten2 = Flatten(name='flatten2')(pool2)
        merged = concatenate([flatten1, flatten2], axis=1)
        activation1 = Activation('relu')(merged)
        dense1 = Dense(1000, kernel_initializer='glorot_normal', name='fc')(activation1)
        activation2 = Activation('relu')(dense1)
        dense2 = Dense(6, name='predictions')(activation2)

        adam = Adam()
        model = Model(inputs=[input_shape], outputs=[dense2])
        model.compile(loss=[vo_loss],
                      optimizer=adam,
                      metrics=[rmse])
        model.summary()

        return model
