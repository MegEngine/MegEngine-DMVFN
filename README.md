# A Dynamic Multi-Scale Voxel Flow Network for Video Prediction (MegEngine implementation)
## Introduction
This project is an official MegEngine implementation of A Dynamic Multi-Scale Voxel Flow Network for Video Prediction.
**Our paper is accepted by CVPR2023.**
## Usage
### Installation

```
git clone https://github.com/MegEngine/MegEngine-DMVFN.git
cd MegEngine-DMVFN
pip3 install -r requirements.txt
```

* Download the pretrained models from [Google Drive](https://drive.google.com/file/d/1hX_J-KsbW2R-um9eEEsgeN1C5Atdan7_/view?usp=sharing).

* Unzip and move the pretrained parameters to pretrained_models/\*

### Installation
Downloads [Cityscapes](https://www.cityscapes-dataset.com/downloads/) and [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php).
### Run

**Train**
```
python3 train_city_vgg.py --train_dataset CityTrainDataset --val_datasets CityValDataset --batch_size 8 --num_gpu 8
```

**Test**
Download [CityValidation](https://drive.google.com/file/d/10zCt-uZFOqgF3tpdhluRqbs-4aScvGR4/view?usp=sharing)
```
python3  test.py --val_datasets CityValDataset --load_path ./pretrained_models/dmvfn_133.pkl
```