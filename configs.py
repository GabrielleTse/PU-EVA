import argparse
import os



def str2bool(x):
    return x.lower() in ('true')

parser = argparse.ArgumentParser()


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', help='train or test [default: train]')
parser.add_argument('--gpu', default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--h5_file', default='dataset/PUGAN_poisson_256_poisson_1024.h5', help='train dataset')
parser.add_argument('--num_point', type=int, default= 256,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=6,   help='Upsampling Ratio [default: 2]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--data_dir', default='data')
parser.add_argument('--augment', type=str2bool, default=True)
parser.add_argument('--restore', action='store_true')
parser.add_argument('--more_up', type=int, default=2)
parser.add_argument('--training_epoch', type=int, default=250)
parser.add_argument('--use_non_uniform', type=str2bool, default=True)
parser.add_argument('--use_random_input', type=str2bool, default=True)
parser.add_argument('--jitter', type=str2bool, default=False)
parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")

parser.add_argument('--patch_num_point', type=int, default= 256)
parser.add_argument('--patch_num_ratio', type=int, default=3)
parser.add_argument('--base_lr_g', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--start_decay_step', type=int, default=50000)
parser.add_argument('--lr_decay_steps', type=int, default=50000)
parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--lr_clip', type=float, default=1e-6)
parser.add_argument('--uniform_w', default=10.0, type=float, help="uniform_weight")
parser.add_argument('--dis_w', default=200.0, type=float, help="fidelity_weight")
parser.add_argument('--gen_update', default=2, type=int, help="gen_update")

parser.add_argument('--use_tb', type=bool, default=False, help='Use tensorboard to log training info')
parser.add_argument('--code_backup', type=bool, default=True, help='code backup')
parser.add_argument('--description', type=str, default="PU-EVA",
                    help='description of current running experiments.')
parser.add_argument('--log_dir', type=str, default="logs")
FLAGS = parser.parse_args()

