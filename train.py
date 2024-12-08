import os
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.emcad.networks import EMCADNet
from trainer import trainer_coca

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/COCA/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='COCA', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

# network related parameters
parser.add_argument('--encoder', type=str,
                    default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--lgag_ks', type=int,
                    default=3, help='Kernel size in LGAG')
parser.add_argument('--activation_mscb', type=str,
                    default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true', 
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true', 
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--no_pretrain', action='store_true', 
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--supervision', type=str,
                    default='last_layer', help='loss supervision: mutation, deep_supervision or last_layer')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    dataset_config = {
        'COCA': {
            'root_path': './data/COCA/train_npz',
            'list_dir': './data/COCA/lists_COCA',
            'num_classes': 4,
            'max_epochs': 1,
            'batch_size': 6,
            'base_lr': 0.00001,
            'img_size': 224,
            'exp_setting': 'default',
        },
    }
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.max_epochs = dataset_config[dataset_name]['max_epochs']
    args.batch_size = dataset_config[dataset_name]['batch_size']
    args.base_lr = dataset_config[dataset_name]['base_lr']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.exp_setting = dataset_config[dataset_name]['exp_setting']

    args.arch = 'EMCAD'
    snapshot_path = f"./model/{args.arch + '_' + args.encoder}/{dataset_name + '_' + str(args.img_size)}/{args.exp_setting}/{'epo' + str(args.max_epochs)}"
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = EMCADNet(num_classes=args.num_classes, 
                   kernel_sizes=args.kernel_sizes, 
                   expansion_factor=args.expansion_factor, 
                   dw_parallel=not args.no_dw_parallel, 
                   add=not args.concatenation, 
                   lgag_ks=args.lgag_ks, 
                   activation=args.activation_mscb, 
                   encoder=args.encoder, 
                   pretrain=not args.no_pretrain).cuda()

    trainer = {'COCA': trainer_coca}
    trainer[dataset_name](args, net, snapshot_path)