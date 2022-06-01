import os

import random
import torch
import torch.backends.cudnn

from config.config import get_config
import data.test_loader
from src.train import Trainer
from src.test import Tester


def main(config):
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
        
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    config.manual_seed = 100
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = None
    if not config.test:
        trainer = Trainer(config)
        trainer.train()
    
        if config.quantize:
            trainer.quantize_model()

    test_loader = data.test_loader.get_loader(config.test_data_path, config.test_batch_size, config.image_size,
                                              shuffle=None, num_workers=int(config.workers))
    tester = Tester(config, test_loader)
    tester.test(net)


if __name__ == "__main__":
    config = get_config()
    main(config)
