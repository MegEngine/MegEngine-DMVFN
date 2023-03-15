# A Dynamic Multi-Scale Voxel Flow Network for Video Prediction
## Introduction
This project is the implement of [A Dynamic Multi-Scale Voxel Flow Network for Video Prediction](https://arxiv.org/abs/).
**Our paper is accepted by CVPR2023.**
[论文中文介绍](https://zhuanlan.zhihu.com/p/)
## Usage
### Installation

```
git clone https://github.com/huxiaotaostasy/DMVFN.git
cd 
pip3 install -r requirements.txt
```

* Download the pretrained models from [Google Drive](). (百度网盘链接: 密码:，把压缩包解开后放在 pretrained_models/\*)

* Unzip and move the pretrained parameters to pretrained_models/\*

### Run

**Train**
```
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py --train_dataset CityTrainDataset --val_datasets CityValDataset --batch_size 8 --num_gpu 8
```

**Test**
```
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 test.py --val_datasets CityValDataset --load_path /data/dmvfn/train_log_CityTrainDataset/DMVFN/dmvfn_299.pkl
```

**Results**

