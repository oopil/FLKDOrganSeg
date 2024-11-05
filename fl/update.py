import sys
sys.path.append("..")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from loss_functions import loss
from tqdm import tqdm
import math
import pdb

# from thop import profile
# from ptflops import get_model_complexity_info
# from ptflops.pytorch_engine import 
import time

# import torch.cuda.profiler as profiler
# import pyprof
# pyprof.init()
# from pthflops import count_ops

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object): # check cost
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).cuda()
        self.loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).cuda()

        self.selected_clients = []
        self.ldr_train = dataset[idxs]
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, amp, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
        if self.args.FP16:
            net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

        epoch_loss = []
        local_eps = math.ceil(self.args.local_min_update / len(self.ldr_train))

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                images = torch.from_numpy(batch['image']).cuda()
                labels = torch.from_numpy(batch['label']).cuda()
                volumeName = batch['name']
                task_ids = batch['task_id']
                net.zero_grad()
                preds = net(images, task_ids)

                term_seg_Dice = self.loss_seg_DICE.forward(preds, labels)
                term_seg_BCE = self.loss_seg_CE.forward(preds, labels) * 10
                term_all = term_seg_Dice + term_seg_BCE

                if self.args.FP16:
                    with amp.scale_loss(term_all, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    term_all.backward()
                optimizer.step()
                batch_loss.append(term_all.item())
                print(f"{iter}/{local_eps}, {batch_idx}/{len(self.ldr_train)}, {term_all:.4f}", end='\r')

                if batch_idx > self.args.local_min_update:
                    break

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateFLKD(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).cuda()
        self.loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).cuda()
        self.loss_seg_Memory = loss.MemoryLoss4MOTS(num_classes=args.num_classes, ignore_index=255).cuda()

        self.alpha = 1.0
        self.selected_clients = []
        self.ldr_train = dataset[idxs]
        self.pretrain = pretrain
        print(f"Alpha value : {self.alpha}")

    def train(self, net, net_old, amp, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)
        if self.args.FP16:
            net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

        epoch_loss = []
        local_eps = math.ceil(self.args.local_min_update / len(self.ldr_train))

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                images = torch.from_numpy(batch['image']).cuda()
                labels = torch.from_numpy(batch['label']).cuda()
                volumeName = batch['name']
                task_ids = batch['task_id']
                net.zero_grad()

                preds, new_outputs = net.forward_all_task(images, task_ids)

                with torch.no_grad():
                    _, old_outputs = net_old.forward_all_task(images, task_ids)

                term_seg_Dice = self.loss_seg_DICE.forward(preds, labels)
                term_seg_BCE = self.loss_seg_CE.forward(preds, labels) * 1
                term_seg_Memory = self.loss_seg_Memory.forward(new_outputs, old_outputs, task_ids) * 1
                term_ori = term_seg_Dice + term_seg_BCE
                term_all = term_seg_Dice + term_seg_BCE + term_seg_Memory * self.alpha

                if self.args.FP16:
                    with amp.scale_loss(term_all, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    term_all.backward()
                optimizer.step()

                batch_loss.append(term_ori.item())
                print(f"{iter}/{local_eps}, {batch_idx}/{len(self.ldr_train)}, {term_ori:.4f}", end='\r')
                
                if batch_idx > self.args.local_min_update:
                    break

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class LocalUpdateFLlocalKD(object):
    def __init__(self, args, dataset=None, idxs=None, pretrained=None, n_sample=3):
        self.args = args
        self.loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).cuda()
        self.loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).cuda()
        self.loss_seg_Memory_global = loss.MemoryLoss4MOTS(num_classes=args.num_classes, ignore_index=255).cuda()
        self.loss_seg_Memory_local = loss.SampledMemoryLoss4MOTS(num_classes=args.num_classes, ignore_index=255).cuda()

        self.selected_clients = []
        self.ldr_train = dataset[idxs]
        self.idx = idxs
        self.pretrained = pretrained
        self.n_sample = n_sample

    def train(self, net, net_old, amp, idx=-1, lr=0.1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum)
        epoch_loss = []
        local_eps = math.ceil(self.args.local_min_update / len(self.ldr_train))

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                images = torch.from_numpy(batch['image']).cuda()
                labels = torch.from_numpy(batch['label']).cuda()
                volumeName = batch['name']
                task_ids = batch['task_id']
                net.zero_grad()

                preds, new_outputs = net.forward_all_task(images, task_ids)

                task_ids = list(set(task_ids.tolist()))
                remaining_task_indices = list(range(7))
                for task_id in task_ids:
                    remaining_task_indices.remove(task_id)

                sample_tids = np.random.choice(remaining_task_indices, self.n_sample, replace=False)
                with torch.no_grad():
                    _, old_outputs = net_old.forward_all_task(images, task_ids)
                    pre_outputs = []
                    for tid in sample_tids:
                        output = self.pretrained[tid](images, task_ids)
                        pre_outputs.append(output)

                term_seg_Dice = self.loss_seg_DICE.forward(preds, labels)
                term_seg_BCE = self.loss_seg_CE.forward(preds, labels)
                term_seg_Memory_global = self.loss_seg_Memory_global.forward(new_outputs, old_outputs, sample_tids)
                term_seg_Memory_local = self.loss_seg_Memory_local.forward(new_outputs, pre_outputs, sample_tids)
                term_ori = term_seg_Dice + term_seg_BCE
                term_all = term_seg_Dice + term_seg_BCE + term_seg_Memory_local + term_seg_Memory_global

                if self.args.FP16:
                    with amp.scale_loss(term_all, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    term_all.backward()
                optimizer.step()

                batch_loss.append(term_ori.item())
                print(f"{iter}/{local_eps}, {batch_idx}/{len(self.ldr_train)}, {term_ori:.4f}", end='\r')
                
                if batch_idx > self.args.local_min_update:
                    break

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


