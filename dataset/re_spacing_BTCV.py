import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pdb import set_trace


spacing = {
    0: [3.0, 1.5, 1.5],
    1: [3.0, 1.5, 1.5],
    2: [3.0, 1.5, 1.5],
    3: [3.0, 1.5, 1.5],
    4: [3.0, 1.5, 1.5],
    5: [3.0, 1.5, 1.5],
    6: [3.0, 1.5, 1.5],
}


spacing_missing_data = []

ori_path = './BTCV'
# new_path = './0123456_spacing_same_modify'
# new_path = './0123456_spacing_same_modify_half'
new_path = './BTCV_modify'

count = -1
for root1, dirs1, _ in os.walk(ori_path):
    for i_dirs1 in tqdm(sorted(dirs1)):  # 0Liver
        print(i_dirs1)
        ###########################################################################
        for root2, dirs2, files2 in os.walk(os.path.join(root1, i_dirs1)):
            for i_dirs2 in sorted(dirs2):  # imagesTr

                for root3, dirs3, files3 in os.walk(os.path.join(root2, i_dirs2)):
                    for i_files3 in sorted(files3):
                        if i_files3[0] == '.':
                            continue
                        # read img
                        print("Processing %s" % (i_files3))
                        img_path = os.path.join(root3, i_files3)
                        imageITK = sitk.ReadImage(img_path)
                        image = sitk.GetArrayFromImage(imageITK)
                        ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
                        ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
                        ori_origin = imageITK.GetOrigin()
                        ori_direction = imageITK.GetDirection()

                        task_id = int(i_dirs1[0])
                        target_spacing = np.array(spacing[task_id])
                        spc_ratio = ori_spacing / target_spacing

                        data_type = image.dtype
                        if i_dirs2 != 'labelsTr':
                            data_type = np.int32

                        if i_dirs2 == 'labelsTr':
                            order = 0
                            mode_ = 'edge'
                        else:
                            order = 3
                            mode_ = 'constant'

                        image = image.astype(np.float)

                        image_resize = resize(image, (int(ori_size[0] * spc_ratio[0]), int(ori_size[1] * spc_ratio[1]),
                                                      int(ori_size[2] * spc_ratio[2])),
                                              order=order, mode=mode_, cval=0, clip=True, preserve_range=True)
                        image_resize = np.round(image_resize).astype(data_type)

                        # save
                        save_path = os.path.join(new_path, i_dirs1, i_dirs2)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        saveITK = sitk.GetImageFromArray(image_resize)
                        saveITK.SetSpacing(target_spacing[[2, 1, 0]])
                        saveITK.SetOrigin(ori_origin)
                        saveITK.SetDirection(ori_direction)
                        # if "volume" in i_files3 or "segmentation" in i_files3:
                        if i_dirs1 == "0Liver":
                            spl = i_files3.split(".")[0]
                            subj_name = spl.split("-")[1]
                            opath = os.path.join(save_path, f"liver_{subj_name}.nii.gz")
                            sitk.WriteImage(saveITK, opath)
                        else:
                            sitk.WriteImage(saveITK, os.path.join(save_path, i_files3))

print("here is the data list which miss spacing information.")
for e in spacing_missing_data:
    print(e)