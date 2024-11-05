# Knowledge distillation based federated learning for partially labeled datasets
This repository includes inplementation of the paper ["Federated Learning with Knowledge Distillation for Multi-Organ Segmentation with Partially Labeled Datasets"](https://www.sciencedirect.com/science/article/pii/S1361841524000811) published in Medical Image Analysis.
This code implementation is based on [DoDNet](https://github.com/jianpengz/DoDNet.git) and [LG-FedAVG](https://github.com/pliang279/LG-FedAvg).

## MOTS Dataset Preparation
Before starting, MOTS should be re-built from the serveral medical organ and tumor segmentation datasets

Partial-label task | Data source
--- | :---:
Liver | [data](https://competitions.codalab.org/competitions/17094)
Kidney | [data](https://kits19.grand-challenge.org/data/)
Hepatic Vessel | [data](http://medicaldecathlon.com/)
Pancreas | [data](http://medicaldecathlon.com/)
Colon | [data](http://medicaldecathlon.com/)
Lung | [data](http://medicaldecathlon.com/)
Spleen | [data](http://medicaldecathlon.com/)

* Download and put these datasets in `dataset/0123456/`. 
* Re-spacing the data by `python re_spacing.py`, the re-spaced data will be saved in `0123456_spacing_same/`. If there are some issues with this code, try `python re_spacing_modify.py`.

The folder structure of dataset should be like

    dataset/0123456_spacing_same/
    ├── 0Liver
    |    └── imagesTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    |    └── labelsTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    ├── 1Kidney
    ├── ...

Voxel spacing information is missing in some CT volumes. Exclude these examples in trainig and testing, i.e., modify MOTS_train.txt and MOTS_test.txt file.
Now you should use MOTS_train_modify.txt and MOTS_test_modify.txt file.
Here is the exception list with different array size between image and label. [file path, image shape, label shape]
```
['./0Liver/imagesTr/liver_48.nii.gz', ]
['./0Liver/imagesTr/liver_49.nii.gz', (611, 611, 423), (640, 640, 169)]
['./0Liver/imagesTr/liver_50.nii.gz', (580, 580, 400), (640, 640, 160)]
['./0Liver/imagesTr/liver_51.nii.gz', (577, 577, 454), (640, 640, 151)]
['./0Liver/imagesTr/liver_52.nii.gz', (535, 535, 395), (640, 640, 158)]
```
These cases looks fine.But it matters in evaluation because the shape of them must be same.
```
['./0Liver/imagesTr/liver_85.nii.gz', (412, 412, 294), (412, 412, 293)]
Hepatic vessels have just 1 voxel difference. maybe fine to use it.
['./2HepaticVessel/imagesTr/hepaticvessel_178.nii.gz', (484, 484, 143), (485, 485, 143)]
['./2HepaticVessel/imagesTr/hepaticvessel_221.nii.gz', (624, 624, 123), (625, 625, 123)]
```
here is the data list which miss spacing information.
```
./0123456/0Liver/imagesTr/volume-28.nii
./0123456/0Liver/imagesTr/volume-29.nii
./0123456/0Liver/imagesTr/volume-30.nii
./0123456/0Liver/imagesTr/volume-31.nii
./0123456/0Liver/imagesTr/volume-32.nii
./0123456/0Liver/imagesTr/volume-33.nii
./0123456/0Liver/imagesTr/volume-34.nii
./0123456/0Liver/imagesTr/volume-35.nii
./0123456/0Liver/imagesTr/volume-36.nii
./0123456/0Liver/imagesTr/volume-37.nii
./0123456/0Liver/imagesTr/volume-38.nii
./0123456/0Liver/imagesTr/volume-39.nii
./0123456/0Liver/imagesTr/volume-40.nii
./0123456/0Liver/imagesTr/volume-41.nii
./0123456/0Liver/imagesTr/volume-42.nii
./0123456/0Liver/imagesTr/volume-43.nii
./0123456/0Liver/imagesTr/volume-44.nii
./0123456/0Liver/imagesTr/volume-45.nii
./0123456/0Liver/imagesTr/volume-46.nii
./0123456/0Liver/imagesTr/volume-47.nii
```

* MOTS_train.txt : original textfile
* MOTS_train_modify.txt : Kidney resampling part is modified and the cases which have different image and label shape were removed.
* MOTS_train_modify_v2.txt : Cases which miss spacing information were excluded.

<!-- ## Model
Pretrained model is available in [checkpoint](https://drive.google.com/file/d/1qj8dJ_G1sHiCmJx_IQjACQhjUQnb4flg/view?usp=sharing) -->

## Training
* federated averaging (FedAVG)
KL divergence loss is not working with binary cross entropy loss. It works with softmax predictions. Implementation is not available in this case.
```
FL=fedavg or flkd
ARCH=multihead or sep_enc or sep_dec or dynconv
EPOCH=1000
ITER=80

time CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --train_list="list/MOTS/MOTS_train_modify_v2.txt" \
  --snapshot_dir="snapshots/${ARCH}_${FL}_${EPOCH}epoch_${ITER}iter" \
  --input_size="64,128,128" \
  --batch_size=2 \
  --num_gpus=1 \
  --num_epochs=${EPOCH} \
  --start_epoch=0 \
  --learning_rate=1e-2 \
  --num_classes=2 \
  --num_workers=8 \
  --weight_std=True \
  --random_mirror=True \
  --random_scale=True \
  --FP16=False \
  --arch=${ARCH} \
  --local_min_update=${ITER} \
  --lr_decay=True \
  --model=${FL}
```

* check tensorboard
> tensorboard --bind_all --logdir=./snapshots --load_fast=false

## Evaluation
You can change input size bigger like '64,256,256'.

```
time CUDA_VISIBLE_DEVICES=${GPU} python evaluate.py \
--val_list="list/MOTS/MOTS_test_modify_v2.txt" \
--reload_from_checkpoint=True \
--reload_path=[PATH_TO_CKPT] \
--ensemble=False \
--save_path="./outputs/${ARCH}_${FL}_${EPOCH}epoch_${ITER}/" \
--input_size="64,128,128" \
--batch_size=1 \
--num_gpus=1 \
--num_workers=2 \
--arch=${ARCH} \
--FP16=False
```

## Post-processing
```
time python ../postp.py --img_folder_path="./outputs/${ARCH}_${FL}_${EPOCH}epoch_${ITER}iter/" | tee -a "./outputs/${ARCH}_${FL}_${EPOCH}epoch_${ITER}iter.txt"
```

## Citation
If this code is helpful for your study, please cite:
```
@article{kim2024federated,
  title={Federated learning with knowledge distillation for multi-organ segmentation with partially labeled datasets},
  author={Kim, Soopil and Park, Heejung and Kang, Myeongkyun and Jin, Kyong Hwan and Adeli, Ehsan and Pohl, Kilian M and Park, Sang Hyun},
  journal={Medical Image Analysis},
  volume={95},
  pages={103156},
  year={2024},
  publisher={Elsevier}
}
```
