import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--decay_epoch', type=int, default=10, help='learning rate decay start epoch num')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--rho', type=float, default=0.95, help='adadelta rho')
parser.add_argument('--eps', type=float, default=1e-7, help='adadelta eps')
parser.add_argument('--decay', type=float, default=2e-5, help='adadelta decay')
parser.add_argument('--dataset_url',
                    default='https://drive.google.com/file/d/1Nw-a13pq_7Q0yHO22UbG4tFHiNci79rz/view?usp=sharing',
                    help='url of dataset')
parser.add_argument('--sample_step', type=int, default=100, help='step of saving sample images')
parser.add_argument('--checkpoint_step', type=int, default=100, help='step of saving checkpoints')
parser.add_argument('--data_path', default='./dataset', help='path to dataset')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--mode', type=str, default='train', help='Trainer mode: train or test')
parser.add_argument('--nf', type=int, default=32, help='The number of filter')
parser.add_argument('--num_test', type=int, default=32, help='The number of test image')

def get_config():
    return parser.parse_args()
