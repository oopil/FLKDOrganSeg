import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from glob import glob
import cv2
from pdb import set_trace

def truncate(CT):
    min_HU = -325
    max_HU = 325
    subtract = 0
    divide = 325.

    # truncate
    CT[np.where(CT <= min_HU)] = min_HU
    CT[np.where(CT >= max_HU)] = max_HU
    CT = CT - subtract
    CT = CT / divide
    return CT


ori_dir = '0123456_spacing_same_modify'
# ori_path = './0123456'
ori_path = f'./{ori_dir}'
new_dir_roi = '0123456_spacing_same_2d_roi'
new_dir_all = '0123456_spacing_same_2d_all'
slice_size = 256

# train_list='list/MOTS/MOTS_train.txt'
# val_list='list/MOTS/MOTS_val.txt'
# data_dir = '../dataset/'
vis_dir = "./vis_2d"
os.makedirs(vis_dir, exist_ok=True)

exception_list = []

count = -1
for root1, dirs1, _ in os.walk(ori_path):
    print(dirs1)
    for i_dirs1 in tqdm(sorted(dirs1)):  # 0Liver
        print(i_dirs1)
        if i_dirs1 not in ['1Kidney']:
            continue
        ###########################################################################
        if i_dirs1 == '1Kidney':
            img_paths = glob(f"../dataset/{ori_dir}/{i_dirs1}/origin/*/imaging.nii.gz")
            for img_path in sorted(img_paths):
                path_spl = img_path.split("/")
                fname = path_spl[-2]
                path_spl[-1] = "segmentation.nii.gz"
                label_path = "/".join(path_spl)
                
                try:
                    imageNII = nib.load(img_path)
                    labelNII = nib.load(label_path)
                except:
                    exception_list.append([img_path, image.shape, label.shape])
                    continue

                imageNII = nib.as_closest_canonical(imageNII)
                labelNII = nib.as_closest_canonical(labelNII)
                image = imageNII.get_data()
                label = labelNII.get_data()
                image = image.astype(np.float)
                label = label.astype(np.int32)

                # no need to transpose because of nib.as_closest_canonical() above
                # image = image.transpose((1, 2, 0)) 
                # label = label.transpose((1, 2, 0))
                image = truncate(image)
                boud_h, boud_w, boud_d = np.where(label >= 1)
                bbx_d_min = boud_d.min()
                bbx_d_max = boud_d.max()
                h,w,d = image.shape

                if image.shape != label.shape:
                    exception_list.append([img_path, image.shape, label.shape])
                    continue

                print(img_path, image.shape)
                image = resize(image, (slice_size, slice_size,d), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
                label = resize(label, (slice_size, slice_size,d), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
                
                for d_idx in range(d):
                    image_slice = image[:,:,d_idx]
                    path_spl = img_path.split("/")
                    path_spl[-5] = new_dir_all
                    os.makedirs("/".join(path_spl),exist_ok=True) # make directory
                    path_spl.append(f"{d_idx}.npy")
                    image_opath = "/".join(path_spl)
                    np.save(image_opath, image_slice)

                    label_slice = label[:,:,d_idx]
                    path_spl = label_path.split("/")
                    path_spl[-5] = new_dir_all
                    os.makedirs("/".join(path_spl),exist_ok=True) # make directory
                    path_spl.append(f"{d_idx}.npy")
                    label_opath = "/".join(path_spl)
                    np.save(label_opath, label_slice)

                    if d_idx >= bbx_d_min and d_idx <= bbx_d_max: # save roi slice
                        path_spl = image_opath.split("/")
                        path_spl[-6] = new_dir_roi
                        os.makedirs("/".join(path_spl[:-1]),exist_ok=True) # make directory
                        image_opath = "/".join(path_spl)
                        np.save(image_opath, image_slice)

                        path_spl = label_opath.split("/")
                        path_spl[-6] = new_dir_roi
                        os.makedirs("/".join(path_spl[:-1]),exist_ok=True) # make directory
                        label_opath = "/".join(path_spl)
                        np.save(label_opath, label_slice)

                    ## visualize
                    if d_idx%30 == 0 and d_idx > 0:
                        # set_trace()
                        image_vis = (image_slice+1)*128
                        label_vis = (label_slice)*128
                        image_vis = np.concatenate([image_vis, label_vis], axis=1)
                        cv2.imwrite(f"{vis_dir}/kidney_{fname}_{d_idx}.png", image_vis.astype(np.uint8))

                    print(f"{d_idx}/{d}", end='\r')

        # if i_dirs1 in ['0Liver', '2HepaticVessel']:
        #     continue

        #############################################################################
        img_paths = glob(f"../dataset/{ori_dir}/{i_dirs1}/imagesTr/*.nii.gz")
        for img_path in sorted(img_paths):
            print(img_path)
            path_spl = img_path.split("/")
            fname = path_spl[-1]
            path_spl[-2] = "labelsTr"
            label_path = "/".join(path_spl)
            imageNII = nib.load(img_path)
            labelNII = nib.load(label_path)
            imageNII = nib.as_closest_canonical(imageNII)
            labelNII = nib.as_closest_canonical(labelNII)
            image = imageNII.get_data()
            label = labelNII.get_data()
            image = image.astype(np.float)
            label = label.astype(np.int32)
            image = truncate(image)
            boud_h, boud_w, boud_d = np.where(label >= 1)
            bbx_d_min = boud_d.min()
            bbx_d_max = boud_d.max()
            h,w,d = image.shape
            print(image.shape)

            if image.shape != label.shape:
                exception_list.append([img_path, image.shape, label.shape])
                continue

            image = resize(image, (slice_size, slice_size,d), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (slice_size, slice_size,d), order=0, mode='edge', cval=0, clip=True, preserve_range=True)

            for d_idx in range(d):

                ## save all slices

                image_slice = image[:,:,d_idx]
                path_spl = img_path.split("/")
                path_spl[-4] = new_dir_all
                os.makedirs("/".join(path_spl),exist_ok=True) # make directory
                path_spl.append(f"{d_idx}.npy")
                image_opath = "/".join(path_spl)
                np.save(image_opath, image_slice)

                label_slice = label[:,:,d_idx]
                path_spl = label_path.split("/")
                path_spl[-4] = new_dir_all
                os.makedirs("/".join(path_spl),exist_ok=True) # make directory
                path_spl.append(f"{d_idx}.npy")
                label_opath = "/".join(path_spl)
                np.save(label_opath, label_slice)

                ## save roi slice

                if d_idx >= bbx_d_min and d_idx <= bbx_d_max:
                    path_spl = image_opath.split("/")
                    path_spl[-5] = new_dir_roi
                    os.makedirs("/".join(path_spl[:-1]),exist_ok=True) # make directory
                    image_opath = "/".join(path_spl)
                    np.save(image_opath, image_slice)

                    path_spl = label_opath.split("/")
                    path_spl[-5] = new_dir_roi
                    os.makedirs("/".join(path_spl[:-1]),exist_ok=True) # make directory
                    label_opath = "/".join(path_spl)
                    np.save(label_opath, label_slice)
                
                ## visualize
                if d_idx%30 == 0 and d_idx > 0:
                    # set_trace()
                    image_vis = (image_slice+1)*127
                    label_vis = (label_slice)*127
                    image_vis = np.concatenate([image_vis, label_vis], axis=1)
                    cv2.imwrite(f"{vis_dir}/{fname}_{d_idx}.png", image_vis.astype(np.uint8))

                print(f"{d_idx}/{d}", end='\r')

print("<<< Here is the exception list with different array size between image and label.>>>")
for e in exception_list:
    print(e)

"""
<<< Here is the exception list with different array size between image and label.>>>
['../dataset/0123456_spacing_same/0Liver/imagesTr/liver_48.nii.gz',]
['../dataset/0123456_spacing_same/0Liver/imagesTr/liver_49.nii.gz', (611, 611, 423), (640, 640, 169)]
['../dataset/0123456_spacing_same/0Liver/imagesTr/liver_50.nii.gz', (580, 580, 400), (640, 640, 160)]
['../dataset/0123456_spacing_same/0Liver/imagesTr/liver_51.nii.gz', (577, 577, 454), (640, 640, 151)]
['../dataset/0123456_spacing_same/0Liver/imagesTr/liver_52.nii.gz', (535, 535, 395), (640, 640, 158)]

This case looks fine.
['../dataset/0123456_spacing_same/0Liver/imagesTr/liver_85.nii.gz', (412, 412, 294), (412, 412, 293)]

Hepatic vessels have just 1 voxel difference. maybe fine to use it.
['../dataset/0123456_spacing_same/2HepaticVessel/imagesTr/hepaticvessel_178.nii.gz', (484, 484, 143), (485, 485, 143)]
['../dataset/0123456_spacing_same/2HepaticVessel/imagesTr/hepaticvessel_221.nii.gz', (624, 624, 123), (625, 625, 123)]

"""