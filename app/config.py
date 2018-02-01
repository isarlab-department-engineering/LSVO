#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

# Network
net_arg = add_argument_group('Network')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_set_dir', type=str, default='/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/Dataset/')
data_arg.add_argument('--kitti_main_dir', type=str, default='KITTI_RGB_dataset')
data_arg.add_argument('--kitti_is_grayscale', type=str2bool, default=False)
data_arg.add_argument('--use_subsampled', type=str2bool, default=True)
data_arg.add_argument('--kitti_of_dir', type=str, default='of_flownet')
data_arg.add_argument('--kitti_of_ext', type=str, default='flo')
data_arg.add_argument('--ram_pre_loading', type=str2bool, default=False)
data_arg.add_argument('--weights_dir', type=str, default='/home/isarlab/PycharmProjects/LSVO/app/weights')
data_arg.add_argument('--kitti_train_dirs', type=eval, nargs='+', default=[])

data_arg.add_argument('--malaga_train_dirs', type=eval, nargs='+', default=[])
data_arg.add_argument('--malaga_main_dir', type=str, default='Malaga_dataset')
data_arg.add_argument('--malaga_is_grayscale', type=str2bool, default=False)
data_arg.add_argument('--malaga_of_dir', type=str, default='optical_flow')
data_arg.add_argument('--malaga_of_ext', type=str, default='flo')

# Training / test parameters
train_arg = add_argument_group('Training')

train_arg.add_argument('--is_train', type=str2bool, default=False)
train_arg.add_argument('--random_seed', type=int, default=123, help='')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
