import numpy as np


class AbstractSample(object):

    def __init__(self, ram_pre_loading):
        self.ram_pre_loading = ram_pre_loading

    def read_features(self):

        raise ("Not implemented - this is an abstract method")

    def read_labels(self):

        raise ("Not implemented - this is an abstract method")


class OpticalFlowSample(AbstractSample):
    def __init__(self, height, width, of, label, ram_pre_loading=False):

        super(OpticalFlowSample, self).__init__(ram_pre_loading)
        self.of_path = of
        self.relative_transform = label
        self.of_height = height
        self.of_width = width
        if self.ram_pre_loading:
            self.features = self.init_features()

    def read_features(self):
        if not self.ram_pre_loading:

            f = open(self.of_path, 'rb')
            magic = np.fromfile(f, np.float32, count=1)
            data2d = None

            if 202021.25 != magic:
                print ('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                data2d = np.fromfile(f, np.float32, count=2 * w * h)
                data2d = np.resize(data2d, (h, w, 2))
            f.close()
            return data2d

        else:
            return self.features

    def init_features(self):
        f = open(self.of_path, 'rb')
        magic = np.fromfile(f, np.float32, count=1)
        data2d = None

        if 202021.25 != magic:
            print ('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data2d = np.fromfile(f, np.float32, count=2 * w * h)
            data2d = np.resize(data2d, (h, w, 2))
        f.close()
        return data2d

    def read_labels(self):
        return self.relative_transform