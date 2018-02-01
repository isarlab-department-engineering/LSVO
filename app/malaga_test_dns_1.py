import tensorflow as tf
import os

from app.config import get_config

from models.trainer import Trainer


if __name__ == "__main__":

    config, unparsed = get_config()

    # MALAGA PARAM

    config.use_Kitti = False
    config.use_Malaga = True

    config.input_width = 224
    config.input_height = 170
    config.input_channel = 1

    config.malaga_test_dirs = ['02', '03', '09']

    # Choose 'multi_task_malaga_dns_1' to test the LS-VO net trained on MALAGA DNS 1 (standard framerate)
    #    or 'single_task_malaga_dns_1' to test the ST-VO net trained on MALAGA DNS 1 (standard framerate)
    config.strategy = 'multi_task_malaga_dns_1'

    if config.strategy == 'multi_task_malaga_dns_1':
        weights_file = os.path.join(config.weights_dir, 'weights_multi_task_malaga_dns_1.hdf5')
    elif config.strategy == 'single_task_malaga_dns_1':
        weights_file = os.path.join(config.weights_dir, 'weights_single_task_malaga_dns_1.hdf5')

    tf.set_random_seed(config.random_seed)
    trainer = Trainer(config)

    trainer.test(weights_file=weights_file)