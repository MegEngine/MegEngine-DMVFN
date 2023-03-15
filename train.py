import os
import cv2
import math
import time
import megengine.distributed as dist
from megengine.data import DataLoader, RandomSampler
import numpy as np
import random
import argparse
from skimage.color import rgb2yuv, yuv2rgb
from util import compute_SSIM
import mge_lpips
import megengine as mge
import logging
import importlib
from tensorboardX import SummaryWriter


from yuv_frame_io import YUV_Read,YUV_Write
from model import Model
from dataset import *
from util import *

def base_build_dataset(name):
    return getattr(importlib.import_module('dataset', package=None), name)()


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return (1e-4 - 1e-5) * mul + 1e-5


def train(model, args):
    step = 0
    nr_eval = args.resume_epoch
    dataset = base_build_dataset(args.train_dataset)
    sampler = RandomSampler(dataset, batch_size=args.batch_size)
    train_data = DataLoader(dataset, sampler=sampler)
    args.step_per_epoch = train_data.__len__()

    step = 0 + args.step_per_epoch * args.resume_epoch

    if dist.get_rank() == 0:
        print('training...')
    time_stamp = time.time()
    for epoch in range(args.resume_epoch, args.epoch):
        # sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu = data
            data_gpu = mge.tensor(data_gpu) / 255. #B,3,C,H,W
            
            
            learning_rate = get_learning_rate(step)

            loss_avg = model.train(data_gpu, learning_rate)
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and dist.get_rank() == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/loss_l1', loss_avg.numpy(), step)
                writer.flush()
            if dist.get_rank() == 0:
                logger.info('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_avg:{:.4e}'.format( \
                    epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, loss_avg.numpy()))
            step += 1
        nr_eval += 1
        # if nr_eval % 1 == 0:
        #     for dataset_name in args.val_datasets:
        #         val_dataset = base_build_dataset(dataset_name)
        #         val_sampler = RandomSampler(val_dataset, batch_size=1, world_size=1, rank=0)
        #         val_data = DataLoader(val_dataset, sampler=val_sampler)
        #         evaluate(model, val_data, dataset_name, nr_eval, step)
        if dist.get_rank() <= 0:    
            model.save_model(save_model_path, epoch, dist.get_rank())   

def evaluate(model, val_data, name, nr_eval, step):
    if name == "CityValDataset" or name == "KittiValDataset" or name == "DavisValDataset":
        lpips_score_mine, psnr_score_mine, ssim_score_mine = np.zeros(5), np.zeros(5), np.zeros(5)
        time_stamp = time.time()
        num = val_data.__len__()
        for i, data in enumerate(val_data):
            data_gpu, _ = data
            data_gpu = mge.tensor(data_gpu) / 255.
            preds = model.eval(data_gpu, name)

            b,n,c,h,w = preds.shape
            assert b==1 and n==5

            gt, pred = data_gpu[0], preds[0]
            for j in range(5):
                psnr = -10 * math.log10(F.mean((gt[j+4] - pred[j]) * (gt[j+4] - pred[j])).detach().numpy())
                ssim_val = compute_SSIM( gt[j+4:j+5], pred[j:j+1]) #(N,)
                x, y = ((gt[j+4:j+5]-0.5)*2.0), ((pred[j:j+1]-0.5)*2.0)
                lpips_val = loss_fn_alex(x, y)

                lpips_score_mine[j] += lpips_val.numpy()
                ssim_score_mine[j] += ssim_val.numpy()
                psnr_score_mine[j] += psnr
                
                gt_1 = (np.transpose(gt[j+4:j+5].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')
                pred_1 = (np.transpose(pred[j:j+1].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')
                if i == 50 and dist.get_rank() == 0:
                        imgs = np.concatenate((gt_1[0], pred_1[0]), 1)[:, :, ::-1]
                        writer_val.add_image(name+str(j) + '/img', imgs.copy(), step, dataformats='HWC')
        eval_time_interval = time.time() - time_stamp

        if dist.get_rank() != 0:
            return
        psnr_score_mine, ssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, lpips_score_mine/num
        for i in range(5):
            logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+
            '%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))

            writer_val.add_scalar(name+' psnr_%d'%(i),  psnr_score_mine[i], step)
            writer_val.add_scalar(name+' ssim_%d'%(i),  ssim_score_mine[i], step)
            writer_val.add_scalar(name+' lpips_%d'%(i),  lpips_score_mine[i], step)
    elif name=="VimeoValDataset":
        lpips_score_mine, ssim_score_mine, psnr_score_mine   = np.zeros(1), np.zeros(1), np.zeros(1)
        time_stamp = time.time()
        num = val_data.__len__()
        for i, data in enumerate(val_data):
            data_gpu, _ = data
            data_gpu = mge.tensor(data_gpu) / 255.
            preds = model.eval(data_gpu, name)

            b,n,c,h,w = preds.shape
            assert b==1 and n==1

            gt, pred = data_gpu[0], preds[0]
            psnr = -10 * math.log10(F.mean((gt[2] - pred[0]) * (gt[2] - pred[0])).detach().numpy())
            ssim_val = compute_SSIM( gt[2:3], pred[0:1] ) #(N,)
            x, y = ((gt[2:3]-0.5)*2.0), ((pred[0:1]-0.5)*2.0)
            lpips_val = loss_fn_alex(x, y)

            lpips_score_mine[0] += lpips_val.numpy()
            ssim_score_mine[0] += ssim_val.numpy()
            psnr_score_mine[0] += psnr
            
            gt_1 = (np.transpose(gt[2:3].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')
            pred_1 = (np.transpose(pred[0:1].detach().numpy(), (0, 2, 3, 1)) * 255).astype('uint8')
            if i == 50 and dist.get_rank() == 0:
                    imgs = np.concatenate((gt_1[0], pred_1[0]), 1)[:, :, ::-1]
                    writer_val.add_image(name+str(0) + '/img', imgs.copy(), step, dataformats='HWC')
        eval_time_interval = time.time() - time_stamp

        if dist.get_rank() != 0:
            return
        psnr_score_mine, ssim_score_mine, lpips_score_mine = psnr_score_mine/num, ssim_score_mine/num, lpips_score_mine/num
        for i in range(1):
            logger.info('%d             '%(nr_eval)+name+'  psnr_%d     '%(i)+'%.4f'%(sum(psnr_score_mine[:(i+1)])/(i+1))+'  ssim_%d     '%(i)+
            '%.4f'%(sum(ssim_score_mine[:(i+1)])/(i+1))+'  lpips_%d     '%(i)+'%.4f'%(sum(lpips_score_mine[:(i+1)])/(i+1)))

            writer_val.add_scalar(name+' psnr_%d'%(i),  psnr_score_mine[i], step)
            writer_val.add_scalar(name+' ssim_%d'%(i),  ssim_score_mine[i], step)
            writer_val.add_scalar(name+' lpips_%d'%(i),  lpips_score_mine[i], step)

@dist.launcher(world_size=8)
def main():   
    rank = dist.get_rank()
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    mge.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--num_gpu', default=4, type=int) # or 8
    parser.add_argument('--batch_size', default=8, type=int, help='minibatch size')
    parser.add_argument('--train_dataset', required=True, type=str, help='CityTrainDataset, KittiTrainDataset, VimeoTrainDataset')
    parser.add_argument('--val_datasets', type=str, nargs='+', default=['CityValDataset'], help='[CityValDataset,KittiValDataset,VimeoValDataset,DavisValDataset]')
    parser.add_argument('--resume_path', default=None, type=str, help='continue to train, model path')
    parser.add_argument('--resume_epoch', default=0, type=int, help='continue to train, epoch')
    global args
    args = parser.parse_args()
    global exp
    exp = os.path.abspath('.').split('/')[-1]
    global loss_fn_alex
    loss_fn_alex = mge_lpips.LPIPS(net='alex')
    global log_path
    log_path = './logs/train_log_{}/{}'.format(args.train_dataset, exp)
    global save_model_path
    save_model_path = './models/train_log_{}/{}'.format(args.train_dataset, exp)

    if dist.get_rank() == 0:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)
        global writer
        writer = SummaryWriter(log_path + '/train')
        global writer_val
        writer_val = SummaryWriter(log_path + '/validate')

    global logger
    logger = logging.getLogger('base')

    model = Model(local_rank=rank, resume_path=args.resume_path, resume_epoch=args.resume_epoch)
    train(model, args)
        
if __name__ == "__main__":
    main()