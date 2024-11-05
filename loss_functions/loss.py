import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum
from pdb import set_trace

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss_avg

class DiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, is_sigmoid=True):
        
        total_loss = []
        if is_sigmoid:
            predict = F.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                # set_trace()
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, \
                        'Expect weight shape [{}], get[{}]'.format(self.num_classes, self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss==total_loss]

        return total_loss.sum()/total_loss.shape[0]


class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(CELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'

        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i])
                ce_loss = torch.mean(ce_loss, dim=[1,2,3])

                ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]

                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]


class SoftmaxDiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, target_idx=0, **kwargs):
        super(SoftmaxDiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.target_idx = target_idx
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, is_sigmoid=True):
        predict = F.softmax(predict, dim=1)
        if self.target_idx in [4,5]:
            tumor = predict[:,1]
            total_loss = self.dice(tumor, target[:,1])
        elif self.target_idx in [6]:
            organ = predict[:,1]
            total_loss = self.dice(organ, target[:,0])
        else:
            organ = predict[:,1]
            tumor = predict[:,2]
            total_loss = (self.dice(organ, target[:,0]) + self.dice(tumor, target[:,1]))/2
            
        return total_loss
        # set_trace()
        # return total_loss.sum()/total_loss.shape[0]


class SoftmaxCELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3,target_idx=0,**kwargs):
        super(SoftmaxCELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.target_idx = target_idx
        self.criterion = nn.BCELoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        # predict: [B,3,w,h,d] or [B,2,w,h,d], target: [B,2,w,h,d]
        predict = F.softmax(predict, dim=1)
        B,_,w,h,d = target.shape
        # set_trace()

        if self.target_idx in [4,5]:
            tumor = predict[:,1]
            total_loss = self.criterion(tumor, target[:,1])
        elif self.target_idx in [6]:
            organ = predict[:,1]
            total_loss = self.criterion(organ, target[:,0])
        else:
            organ = predict[:,1]
            tumor = predict[:,2]
            total_loss = (self.criterion(organ, target[:,0]) + self.criterion(tumor, target[:,1]))/2
            
        return total_loss.mean()

        bg = 1-target.sum(dim=1, keepdim=True)
        if self.target_idx in [4,5]: # only tumor label
            onehot_target = torch.cat([bg,target[:,1:2,...]], dim=1)
        else:
            onehot_target = torch.cat([bg,target], dim=1)
        new_target = torch.argmax(onehot_target, dim=1)
        total_loss = self.criterion(predict, new_target).mean(dim=(1,2,3))
        return total_loss.sum()/total_loss.shape[0]


class SingleDiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, target_idx=0, **kwargs):
        super(SingleDiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.target_idx = target_idx
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, is_sigmoid=True):
        predict = F.sigmoid(predict)
        total_loss = self.dice(predict[:,0], target[:,0])
        return total_loss


class SingleCELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3,target_idx=0,**kwargs):
        super(SingleCELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.target_idx = target_idx
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        # self.criterion = nn.BCELoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        # predict: [B,3,w,h,d] or [B,2,w,h,d], target: [B,2,w,h,d]
        # predict = F.sigmoid(predict)
        # B,_,w,h,d = target.shape
        # set_trace()
        total_loss = self.criterion(predict[:,0], target[:,0])
        return total_loss.mean()

class SingleMemoryLoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(SingleMemoryLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        self.num_tasks = 4
        print(f"Task Memory Loss Temperature ... {self.temperature}")
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def forward(self, outputs, old_outputs, task_ids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(self.num_tasks))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        # set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            with torch.no_grad():
                target = torch.sigmoid(old_outputs[task_id][:, 0]*self.temperature)
            ce_loss = self.criterion(outputs[task_id][:, 0], target)
            ce_loss = torch.mean(ce_loss, dim=[1,2,3])

            ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]

            total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]
        return total_loss.sum()/total_loss.shape[0]

        
class MemoryLoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(MemoryLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        print(f"Task Memory Loss Temperature ... {self.temperature}")
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights
    
    def set_temperature(self, T):
        self.temperature = T

    def forward(self, outputs, old_outputs, task_ids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(7))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        # set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            for i in range(self.num_classes):
                if [task_id, i] in self.exception_list:
                    continue

                if i != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[task_id][:, i]*self.temperature)
                    ce_loss = self.criterion(outputs[task_id][:, i], target)
                    ce_loss = torch.mean(ce_loss, dim=[1,2,3])

                    ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]

                    total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        # return total_loss.sum()
        return total_loss.sum()/total_loss.shape[0]

class MemoryLoss4MOTS_3Client(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(MemoryLoss4MOTS_3Client, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        print(f"Task Memory Loss Temperature ... {self.temperature}")
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights
    
    def set_temperature(self, T):
        self.temperature = T

    def forward(self, outputs, old_outputs, task_ids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(3))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        # set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            for i in range(self.num_classes):
                if [task_id, i] in self.exception_list:
                    continue

                if i != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[task_id][:, i]*self.temperature)
                    ce_loss = self.criterion(outputs[task_id][:, i], target)
                    ce_loss = torch.mean(ce_loss, dim=[1,2,3])

                    ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]

                    total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        # return total_loss.sum()
        return total_loss.sum()/total_loss.shape[0]


class SampledMemoryLoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(SampledMemoryLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        print(f"Task Memory Loss Temperature ... {self.temperature}")
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def forward(self, outputs, old_outputs, sample_tids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        # set_trace()
        ## calculate loss between old pred and new pred
        for i, tid in enumerate(sample_tids):
            for cid in range(self.num_classes):
                if [tid, cid] in self.exception_list:
                    continue

                if cid != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[i][:, cid]*self.temperature)
                    ce_loss = self.criterion(outputs[tid][:, cid], target)
                    ce_loss = torch.mean(ce_loss, dim=[1,2,3])
                    ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]
                    total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        # return total_loss.sum()
        return total_loss.sum()/total_loss.shape[0]


class OrganKDLoss4MOTS(nn.Module): # only for organ seg
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(OrganKDLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        print(f"Task Memory Loss Temperature ... {self.temperature}")

    def forward(self, outputs, old_outputs, sample_tids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        # set_trace()
        ## calculate loss between old pred and new pred
        for i, tid in enumerate(sample_tids):
            with torch.no_grad():
                target = torch.sigmoid(old_outputs[i][:, 0]*self.temperature)
            ce_loss = self.criterion(outputs[tid][:, 0], target)
            ce_loss = torch.mean(ce_loss, dim=[1,2,3])
            ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]
            total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        # return total_loss.sum()
        return total_loss.sum()/total_loss.shape[0]


class SampledOrganKDLoss4MOTS(nn.Module): 
    # only for organ seg, distilled image is used only for one local model
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(SampledOrganKDLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        print(f"Task Memory Loss Temperature ... {self.temperature}")

    def forward(self, outputs, old_outputs, distill_tids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        # set_trace()
        ## calculate loss between old pred and new pred
        for bid, tid in enumerate(distill_tids):
            with torch.no_grad():
                target = torch.sigmoid(old_outputs[tid][bid:bid+1, 0]*self.temperature)
            ce_loss = self.criterion(outputs[tid][bid:bid+1, 0], target)
            ce_loss = torch.mean(ce_loss, dim=[1,2,3])
            ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]
            total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        # return total_loss.sum()
        return total_loss.sum()/total_loss.shape[0]



# does not work in sigmoid activation. works with softmax
class KLMemoryLoss4MOTS(nn.Module): 
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(KLMemoryLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction='none', log_target=False)
        # self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        # print(f"Task Memory Loss Temperature ... {self.temperature}")
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 
    
    def set_temperature(self, T):
        self.temperature = T

    def forward(self, outputs, old_outputs, task_ids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(7))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            for i in range(self.num_classes):
                if [task_id, i] in self.exception_list:
                    continue

                if i != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[task_id][:, i]*self.temperature)
                    pred = torch.sigmoid(outputs[task_id][:, i])
                    kldiv = (target*(target.log()-pred.log()))
                    kldiv = self.criterion(pred.log(), target)

                    kldiv = torch.mean(kldiv, dim=[1,2,3])

                    kldiv_avg = kldiv[target[:, 0, 0, 0] != -1].sum() / kldiv[target[:, 0, 0, 0] != -1].shape[0]

                    total_loss.append(kldiv_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        # return total_loss.sum()
        return total_loss.sum()/total_loss.shape[0]


class SelectiveMemoryLoss4MOTS(nn.Module):
    def __init__(self, level = 0.7, ignore_index=None,num_classes=3, **kwargs):
        super(SelectiveMemoryLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.select_level = level
        # v1 0.70
        print(f"Selective knowledge distillation loss ... level {self.select_level}")
        self.temperature = 1.0
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def forward(self, outputs, old_outputs, task_ids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(7))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        # set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            for i in range(self.num_classes):
                if [task_id, i] in self.exception_list:
                    continue

                if i != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[task_id][:, i]*self.temperature)
                        select_map = torch.clamp(torch.gt(target, self.select_level).sum((1,2,3)), 0, 1)
                    
                    ce_loss = self.criterion(outputs[task_id][:, i], target)
                    ce_loss = torch.mean(ce_loss, dim=[1,2,3])
                    ce_loss_avg = (ce_loss*select_map)/(select_map.sum()+1e-10)

                    total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]


class IncrementalMemoryLoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(IncrementalMemoryLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.temperature = 1.0
        print(f"Incremental Task Memory Loss Temperature ... {self.temperature}")
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 
    
    def set_temperature(self, T):
        self.temperature = T

    def forward(self, outputs, old_outputs, task_ids, ignore_tid):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(7))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        # set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            for i in range(self.num_classes):
                if [task_id, i] in self.exception_list or task_id >= ignore_tid:
                    continue

                if i != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[task_id][:, i]*self.temperature)
                    ce_loss = self.criterion(outputs[task_id][:, i], target)
                    ce_loss = torch.mean(ce_loss, dim=[1,2,3])

                    ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]

                    total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]



class MemoryEnhancingLoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(MemoryEnhancingLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        # v4: 1.1/3.5, v5: 2.0/4.0
        self.temperature = 1.1
        self.uncertain_margin = 3.5
        print(f"Incremental Task Memory Loss Temperature ... {self.temperature}, uncertain margin ... {self.uncertain_margin}")
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(2):
            voxels_i = [mask == i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum / voxels_i).astype(np.float32)
            weights = torch.where(mask == i, w_i * torch.ones_like(weights).float(), weights)

        return weights
    
    def set_temperature(self, T):
        self.temperature = T

    def forward(self, outputs, old_outputs, task_ids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(7))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        # set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            for i in range(self.num_classes):
                if [task_id, i] in self.exception_list:
                    continue

                if i != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[task_id][:, i])
                        ## gaussian filtering?
                        ## remove uncertain predictions in loss computation
                        certain_mask = torch.lt(target, 0.5-self.uncertain_margin) \
                            + torch.gt(target, 0.5+self.uncertain_margin)

                        ## enhance predictions for other organs. temperature > 1 here
                        target_enhance = torch.sigmoid(old_outputs[task_id][:, i]*self.temperature)


                    ## target memory enhancing loss (semi-supervision)
                    ce_loss2 = self.criterion(outputs[task_id][:, i], target_enhance)*certain_mask
                    ce_loss2 = torch.mean(ce_loss2, dim=[1,2,3])
                    memory_enhancing_loss_avg = ce_loss2[target[:, 0, 0, 0] != -1].sum() / ce_loss2[target[:, 0, 0, 0] != -1].shape[0]

                    # ce_loss_avg = memory_loss_avg + memory_enhancing_loss_avg
                    # if torch.isnan(ce_loss_avg):
                    #     set_trace()

                    total_loss.append(memory_enhancing_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]


class TaskAdaptiveLoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(TaskAdaptiveLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.NLLLoss(reduction='mean', ignore_index=ignore_index)

    def forward(self, output, target, task_ids):
        # set_trace()
        # assert output.shape == target.shape, 'output & target shape do not match'
        task_ids = task_ids.tolist()
        # output = F.softmax(output, dim=1)
        output = torch.clamp(output, min=1e-10, max=1) # [B,14+1,w,h,d]
        outputs = []
        for bid, tid in enumerate(task_ids):
            output_task = output[bid:bid+1,2*tid:2*tid+2]
            output_bg = torch.cat([output[bid:bid+1,:2*tid],output[bid:bid+1,2*tid+2:]], dim=1).sum(1,keepdim=True)
            output_p = torch.cat([output_bg, output_task], 1)
            outputs.append(output_p)
        
        outputs = torch.cat(outputs, dim=0)
        # binary labels to multi-class label
        # target [B,2,w,h,d]
        bg = torch.ones_like(target[:,0:1])*0.1
        target = torch.cat([bg,target],dim=1)
        multiclass_label = torch.argmax(target, dim=1)
        total_loss = self.criterion(torch.log(outputs), multiclass_label)
        return total_loss

    def forward_save(self, output, target, task_ids):
        set_trace()
        # assert output.shape == target.shape, 'output & target shape do not match'
        task_ids = task_ids.tolist()
        # output = F.softmax(output, dim=1)
        output = torch.clamp(output, min=1e-10, max=1)
        output = torch.split(output,split_size_or_sections=2, dim=1) # [B,2,w,h,d] * 7 + [B,2,w,h,d]
        outputs = []
        for bid, tid in enumerate(task_ids):
            remain_tids = list(range(7))
            remain_tids.remove(tid)
            output_task = output[tid][bid:bid+1]
            output_bg = torch.cat([output[i][bid:bid+1] for i in remain_tids] + [output[-1][bid:bid+1]],\
                dim=0).sum(0,keepdim=True).sum(1,keepdim=True)
            # output_bg = torch.sum([e[bid:bid+1] for i,e in enumerate(output) if i in remain_tids])
            output_p = torch.cat([output_bg, output_task], 1)
            outputs.append(output_p)
        
        outputs = torch.cat(outputs, dim=0)
        # binary labels to multi-class label
        # target [B,2,w,h,d]
        bg = torch.ones_like(target[:,0:1])*0.1
        target = torch.cat([bg,target],dim=1)
        multiclass_label = torch.argmax(target, dim=1)
        total_loss = self.criterion(torch.log(outputs), multiclass_label)
        return total_loss


class ExclusionLoss(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(ExclusionLoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def forward(self, outputs, old_outputs, task_ids):
        # assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        ## find remaining tasks
        task_ids = list(set(task_ids.tolist()))
        remaining_task_indices = list(range(7))
        for task_id in task_ids:
            remaining_task_indices.remove(task_id)

        # set_trace()
        ## calculate loss between old pred and new pred
        for task_id in remaining_task_indices:
            for i in range(self.num_classes):
                if [task_id, i] in self.exception_list:
                    continue

                if i != self.ignore_index:
                    with torch.no_grad():
                        target = torch.sigmoid(old_outputs[task_id][:, i])
                    ce_loss = self.criterion(outputs[task_id][:, i], target)
                    ce_loss = torch.mean(ce_loss, dim=[1,2,3])

                    ce_loss_avg = ce_loss[target[:, 0, 0, 0] != -1].sum() / ce_loss[target[:, 0, 0, 0] != -1].shape[0]

                    total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]


class FTDiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(FTDiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 

    def forward(self, predict, target, outputs_old, task_ids):
        # set_trace()
        ## psuedo label preparation
        with torch.no_grad():
            outputs_old = torch.cat(outputs_old, dim=1) # [B,7*2,w,h,d]
            outputs_old = F.sigmoid(outputs_old)
            for bid,tid in enumerate(task_ids.tolist()):
                outputs_old[bid:bid+1,2*tid:2*tid+2] = target[bid:bid+1]
            binary_pl_round = torch.round(outputs_old).clamp(0,1)
            pl_compare = torch.argmax(outputs_old, dim=1).unsqueeze(1)
            binary_pl_compare = torch.zeros_like(pl_compare).repeat(1,7*2,1,1,1)
            binary_pl_compare = binary_pl_compare.scatter(1, pl_compare, 1) # [B,2*7,w,h,d]
            binary_pl = binary_pl_round * binary_pl_compare
            # binary_pl = torch.split(binary_pl,split_size_or_sections=2, dim=1)  # [B,2,w,h,d] * 7

        total_loss = []
        predict = F.sigmoid(predict)

        total_loss = []
        ## compute total loss
        for _,tid in enumerate(task_ids.tolist()):
            for cid in range(self.num_classes):
                if [tid, cid] not in self.exception_list and cid != self.ignore_index:
                    dice_loss = self.dice(predict[:, cid], binary_pl[:, cid])
                    total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]


class FTCELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None,num_classes=3, **kwargs):
        super(FTCELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.exception_list = [[4,0],[5,0],[6,1]] # [task_id, cls_id] 


    def forward(self, predict, target, outputs_old, task_ids):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        # set_trace()
        ## psuedo label preparation
        with torch.no_grad():
            outputs_old = torch.cat(outputs_old, dim=1) # [B,7*2,w,h,d]
            outputs_old = F.sigmoid(outputs_old)
            for bid,tid in enumerate(task_ids.tolist()):
                outputs_old[bid:bid+1,2*tid:2*tid+2] = target[bid:bid+1]
            binary_pl_round = torch.round(outputs_old).clamp(0,1)
            pl_compare = torch.argmax(outputs_old, dim=1).unsqueeze(1)
            binary_pl_compare = torch.zeros_like(pl_compare).repeat(1,7*2,1,1,1)
            binary_pl_compare = binary_pl_compare.scatter(1, pl_compare, 1) # [B,2*7,w,h,d]
            binary_pl = binary_pl_round# * binary_pl_compare
        
        total_loss = []
        ## compute total loss
        for _,tid in enumerate(task_ids.tolist()):
            for cid in range(self.num_classes):
                if [tid, cid] not in self.exception_list and cid != self.ignore_index:
                    ce_loss = self.criterion(predict[:, cid], binary_pl[:, cid])
                    ce_loss = torch.mean(ce_loss, dim=[1,2,3])
                    ce_loss_avg = ce_loss[target[:, cid, 0, 0, 0] != -1].sum() / ce_loss[target[:, cid, 0, 0, 0] != -1].shape[0]
                    total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]

        return total_loss.sum()/total_loss.shape[0]
