import os
import cv2
import math
import time
import numpy as np
import random
import argparse
from skimage.color import rgb2yuv, yuv2rgb
from util import compute_SSIM
from megengine.data import DataLoader, RandomSampler
import mge_lpips
import megengine as mge
import logging
import importlib


from yuv_frame_io import YUV_Read,YUV_Write
from model import Model
from dataset import *
from util import *

seed = 1234
random.seed(seed)
np.random.seed(seed)
mge.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--save_img', action='store_true', help='save or not')
parser.add_argument('--val_datasets', type=str, nargs='+', default=['CityValDataset'], help='[CityValDataset,KittiValDataset,VimeoValDataset,DavisValDataset]')
parser.add_argument('--load_path', required=True, type=str, help='model path')
args = parser.parse_args()

exp = os.path.abspath('.').split('/')[-1]
loss_fn_alex = mge_lpips.LPIPS(net='alex')
log_path = './logs/test_log/{}'.format(exp)


if not os.path.exists(log_path):
    os.makedirs(log_path)
setup_logger('base', log_path, 'test', level=logging.INFO, screen=True, to_file=True)
logger = logging.getLogger('base')


def base_build_dataset(name):
    return getattr(importlib.import_module('dataset', package=None), name)()

def test(model, save_img=False):
    step = 0
    nr_eval = 0
    print('testing...')

    for dataset_name in args.val_datasets:
        val_dataset = base_build_dataset(dataset_name)
        val_sampler = RandomSampler(val_dataset, batch_size=1, world_size=1, rank=0)
        val_data = DataLoader(val_dataset, num_workers=1, sampler=val_sampler)
        evaluate(model, val_data, dataset_name, nr_eval, step, save_img)


def evaluate(model, val_data, name, nr_eval, step, save_img):
    save_img_path = './save_img/test_log_{}/{}'.format(name, exp)
    if name == "CityValDataset" or name == "KittiValDataset" or name == "DavisValDataset":
        lpips_score_mine, psnr_score_mine, ssim_score_mine = np.zeros(5), np.zeros(5), np.zeros(5)
        time_stamp = time.time()
        num = val_data.__len__()
        for i, data in enumerate(val_data):
            data_gpu, data_name = data
            data_gpu = mge.tensor(data_gpu) / 255.
            preds = model.eval(data_gpu, name)

            b,n,c,h,w = preds.shape
            assert b==1 and n==5

            gt, pred = data_gpu[0], preds[0]
            if save_img:
                pred_1_name = os.path.join(save_img_path, data_name[0])
                print(pred_1_name, data_name)
                if not os.path.exists(pred_1_name):
                    os.makedirs(pred_1_name)

            for j in range(5):
                psnr = -10 * math.log10(F.mean((gt[j+4] - pred[j]) * (gt[j+4] - pred[j])).detach().numpy())
                ssim_val = compute_SSIM( gt[j+4:j+5], pred[j:j+1]) # return (N,)
                x, y = ((gt[j+4:j+5]-0.5)*2.0), ((pred[j:j+1]-0.5)*2.0)
                lpips_val = loss_fn_alex(x, y)

                lpips_score_mine[j] += lpips_val.numpy()
                ssim_score_mine[j] += ssim_val.numpy()
                psnr_score_mine[j] += psnr
                
                gt_1 = (np.transpose(gt[j+4:j+5].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')
                pred_1 = (np.transpose(pred[j:j+1].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')

                if save_img:
                    cv2.imwrite(os.path.join(pred_1_name, 'pred_%d.png'%(j+1)), pred_1[0])

        eval_time_interval = time.time() - time_stamp

        psnr_score_mine, ssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, lpips_score_mine/num
        for i in range(5):
            logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+
            '  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))
    elif name=="VimeoValDataset":
        lpips_score_mine, psnr_score_mine, ssim_score_mine = np.zeros(1), np.zeros(1), np.zeros(1)

        time_stamp = time.time()
        num = val_data.__len__()
        for i, data in enumerate(val_data):
            data_gpu, data_name = data
            data_gpu = mge.tensor(data_gpu) / 255.
            preds = model.eval(data_gpu, name)

            b,n,c,h,w = preds.shape
            assert b==1 and n==1

            gt, pred = data_gpu[0], preds[0]
            if save_img:
                pred_1_name = os.path.join(save_img_path, data_name[0])
                print(pred_1_name, data_name)
                if not os.path.exists(pred_1_name):
                    os.makedirs(pred_1_name)

            
            gt, pred = data_gpu[0], preds[0]
            psnr = -10 * math.log10(F.mean((gt[2] - pred[0]) * (gt[2] - pred[0])).detach().numpy())
            ssim_val = compute_SSIM( gt[2:3], pred[0:1]) # return (N,)
            x, y = ((gt[2:3]-0.5)*2.0), ((pred[0:1]-0.5)*2.0)
            lpips_val = loss_fn_alex(x, y)

            lpips_score_mine[0] += lpips_val.numpy()
            ssim_score_mine[0] += ssim_val.numpy()
            psnr_score_mine[0] += psnr
            
            gt_1 = (np.transpose(gt[2:3].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')
            pred_1 = (np.transpose(pred[0:1].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')

            if save_img:
                cv2.imwrite(os.path.join(pred_1_name, 'pred_%d.png'%(j+1)), pred_1[0])

        eval_time_interval = time.time() - time_stamp

        psnr_score_mine, ssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, lpips_score_mine/num
        for i in range(1):
            logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+'%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+
            '  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))



   
if __name__ == "__main__":    
    model = Model(local_rank=0, load_path=args.load_path, simply_infer=True)
    test(model, args.save_img)  
