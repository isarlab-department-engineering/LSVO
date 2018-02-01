import tensorflow as tf
import os

from app.config import get_config

from models.trainer import Trainer


if __name__ == "__main__":

    config, unparsed = get_config()

    # KITTI PARAM

    config.use_Kitti = True
    config.use_Malaga = False

    config.input_width = 300
    config.input_height = 94
    config.input_channel = 1

    config.kitti_test_dirs = ['08', '09', '10']

    # Choose 'multi_task_kitti_dns_1' to test the LS-VO net trained on KITTI DNS 1 (standard framerate)
    #    or 'single_task_kitti_dns_1' to test the ST-VO net trained on KITTI DNS 1 (standard framerate)
    config.strategy = 'multi_task_kitti_dns_1'

    if config.strategy == 'multi_task_kitti_dns_1':
        weights_file = os.path.join(config.weights_dir, 'weights_multi_task_kitti_dns_1.hdf5')
    elif config.strategy == 'single_task_kitti_dns_1':
        weights_file = os.path.join(config.weights_dir, 'weights_single_task_kitti_dns_1.hdf5')

    tf.set_random_seed(config.random_seed)
    trainer = Trainer(config)

    trainer.test(weights_file=weights_file)