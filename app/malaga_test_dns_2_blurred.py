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

    config.malaga_test_dirs = ['02_dns_2_blur_10', '03_dns_2_blur_10', '09_dns_2_blur_10']

    # Choose 'multi_task_malaga_dns_2_blurred' to test the LS-VO net trained on MALAGA DNS 2 (framerate: 5Hz)
    #    or 'single_task_malaga_dns_2_blurred' to test the ST-VO net trained on MALAGA DNS 2 (framerate: 5Hz)
    config.strategy = 'multi_task_malaga_dns_2_blurred'

    if config.strategy == 'multi_task_malaga_dns_2_blurred':
        weights_file = os.path.join(config.weights_dir, 'weights_multi_task_malaga_dns_2.hdf5')
    elif config.strategy == 'single_task_malaga_dns_2_blurred':
        weights_file = os.path.join(config.weights_dir, 'weights_single_task_malaga_dns_2.hdf5')

    tf.set_random_seed(config.random_seed)
    trainer = Trainer(config)

    trainer.test(weights_file=weights_file)