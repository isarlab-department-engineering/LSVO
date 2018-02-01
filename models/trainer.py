from models.MultiTaskModel import MultiTaskModel
from models.SingleTaskModel import SingleTaskModel


class Trainer(object):
    def __init__(self, config):

        self.config = config

        if self.config.strategy == 'single_task_kitti_dns_1' \
                or self.config.strategy == 'single_task_kitti_dns_2'\
                or self.config.strategy == 'single_task_kitti_dns_2_blurred'\
                or self.config.strategy == 'single_task_malaga_dns_1'\
                or self.config.strategy == 'single_task_malaga_dns_2'\
                or self.config.strategy == 'single_task_malaga_dns_2_blurred':
            self.model = SingleTaskModel(config)
        elif self.config.strategy == 'multi_task_kitti_dns_1' \
                or self.config.strategy == 'multi_task_kitti_dns_2'\
                or self.config.strategy == 'multi_task_kitti_dns_2_blurred'\
                or self.config.strategy == 'multi_task_malaga_dns_1'\
                or self.config.strategy == 'multi_task_malaga_dns_2'\
                or self.config.strategy == 'multi_task_malaga_dns_2_blurred':
            self.model = MultiTaskModel(config)

    def test(self, weights_file):
        self.model.test(weights_file=weights_file)
