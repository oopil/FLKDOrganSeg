import os, sys
import copy
import argparse
import os.path as osp

sys.path.append("..")

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import unet3D_multihead
import unet3D_sep_dec
import unet3D_sep_enc
import unet3D_DynConv882

from MOTSDataset_sep import MOTSDataSet, my_collate
import update
import random
import timeit
from tensorboardX import SummaryWriter
from loss_functions import loss
from sklearn import metrics
from math import ceil
from engine import Engin
from apex import amp
from apex.parallel import convert_syncbn_model

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

    parser = argparse.ArgumentParser(description="unet3D_multihead")

    parser.add_argument("--model", type=str, default='fedavg', help="fedavg, tml, lwf")
    parser.add_argument("--arch", type=str, default='multihead', help="multihead, dynconv, sep_dec, sep_enc")
    parser.add_argument("--data_dir", type=str, default='/data/soopil/DoDNet/dataset/')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/fold1/')
    parser.add_argument("--reload_path", type=str, default='snapshots/fold1/xx.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='64,64,64')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')

    # federated arguments
    # parser.add_argument('--epochs', type=int, default=1000, help="rounds of training")
    parser.add_argument("--cost_eval", type=str2bool, default=False)
    parser.add_argument('--num_users', type=int, default=7, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_min_update', type=int, default=80, help="the number of local update")
    parser.add_argument('--local_bs', type=int, default=2, help="local batch size: B")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--local_ep_pretrain', type=int, default=0, help="the number of pretrain local ep")
    parser.add_argument('--lr_decay', type=str2bool, default=True, help="learning rate decay")
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if args.arch == 'multihead':
            net_glob = unet3D_multihead.UNet3D(num_classes=args.num_classes, weight_std=args.weight_std)
        elif args.arch == 'dynconv':
            net_glob = unet3D_DynConv882.UNet3D(num_classes=args.num_classes, weight_std=args.weight_std)
        elif args.arch == 'sep_enc':
            net_glob = unet3D_sep_enc.UNet3D(num_classes=args.num_classes, weight_std=args.weight_std)
        elif args.arch == 'sep_dec':
            net_glob = unet3D_sep_dec.UNet3D(num_classes=args.num_classes, weight_std=args.weight_std)
        else:
            assert False, "Not a proper model name."

        net_glob.train()

        device = torch.device('cuda:{}'.format(args.local_rank))
        net_glob.to(device)

        optimizer = torch.optim.SGD(net_glob.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if args.num_gpus > 1:
            net_glob = engine.data_parallel(net_glob)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    net_glob.load_state_dict(checkpoint['net_glob'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    net_glob.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        train_datasets = [MOTSDataSet(args.data_dir, \
                        args.train_list, 
                        max_iters=args.itrs_each_epoch * args.batch_size, 
                        crop_size=input_size, 
                        scale=args.random_scale, 
                        mirror=args.random_mirror,
                        cost_eval=args.cost_eval,
                        target_task=i) for i in range(args.num_users)]
        
        train_loaders = [torch.utils.data.DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        drop_last=False,
                        shuffle=True,
                        pin_memory=True,
                        sampler=None,
                        collate_fn=my_collate) for train_dataset in train_datasets]

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None
        val_best_loss = 999999

        loss_train = []
        net_best = None
        best_loss = None
        best_acc = None
        best_epoch = None

        lr = args.learning_rate
        results = []

        for epoch in range(args.num_epochs):
            w_glob = None
            loss_locals = []
            m = args.num_users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            if args.lr_decay:
                lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)
            print("Epoch {}, lr: {:.6f}, {}".format(epoch, lr, idxs_users))

            for idx in idxs_users:
                if args.model == 'fedavg':
                    local = update.LocalUpdate(args=args, dataset=train_loaders, idxs=idx)
                elif args.model == 'flkd':
                    local = update.LocalUpdateFLKD(args=args, dataset=train_loaders, idxs=idx)
                elif args.model == 'fllocalkd':
                    local = update.LocalUpdateFLlocalKD(\
                        args=args, 
                        dataset=train_loaders, 
                        idxs=idx, 
                        pretrained=pretrained_models, # need to add code for loading local models
                        n_sample=1)
                else:
                    assert False, f"Not a proper model name. {args.model}"
                net_local = copy.deepcopy(net_glob)

                if args.model in ['lwf', 'lwf_select', 'kldiv', 'tmel']:
                    w_local, loss = local.train(net=net_local.to(device), net_old=net_glob, amp=amp, lr=lr)
                else:
                    w_local, loss = local.train(net=net_local.to(device), amp=amp, lr=lr)

                loss_locals.append(copy.deepcopy(loss))

                if w_glob is None:
                    w_glob = copy.deepcopy(w_local)
                else:
                    for k in w_glob.keys():
                        w_glob[k] += w_local[k]

                if (epoch % 100) == 0:
                    torch.save(net_local.state_dict(),osp.join(args.snapshot_dir, f'task{idx}' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))


            # update global weights
            for k in w_glob.keys():
                w_glob[k] = torch.div(w_glob[k], m)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
            print('Epoch {:3d} / {:3d}, Average loss {:.4f}'.format(epoch, args.num_epochs, loss_avg))

            if (args.local_rank == 0):
                # print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
                #                                                           epoch_loss.item()))
                # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss', loss_avg, epoch)

            if (epoch > 0) and (args.local_rank == 0) and (epoch % 100 == 0):
                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': net_glob.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_multihead_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
                else:
                    torch.save(net_glob.state_dict(),osp.join(args.snapshot_dir, 'MOTS_multihead_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))

            if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': net_glob.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_multihead_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                else:
                    torch.save(net_glob.state_dict(),osp.join(args.snapshot_dir, 'MOTS_multihead_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                break

        print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))


if __name__ == '__main__':
    main()
