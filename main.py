import os

from config.config import get_config
from data.dataloader import get_loader
from src.train import Trainer
from util.util import download_url, unzip_zip_file


def main(config):
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # config.manual_seed = random.randint(1, 10000)
    # print("Random Seed: ", config.manual_seed)
    # random.seed(config.manual_seed)
    # torch.manual_seed(config.manual_seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(config.manual_seed)

    # cudnn.benchmark = True

    if not os.path.exists(config.data_path):
        download_url(config.dataset_url, 'datasets.zip')
        unzip_zip_file('datasets.zip', config.data_path)

    data_loader = get_loader(config.data_path, config.batch_size, config.image_size,
                            shuffle=True, num_workers=int(config.workers))

    trainer = Trainer(config, data_loader)
    trainer.train()


if __name__ == "__main__":
    config = get_config()
    main(config)
