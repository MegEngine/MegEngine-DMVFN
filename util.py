import cv2
import scipy.misc
from io import BytesIO
import logging
import numpy as np
import os
from tensorboardX.summary import Summary
import matplotlib.colors as cl
from datetime import datetime
import megengine.functional as F
from megengine.functional import exp
import megengine
import megengine as mge

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, save_dir, phase, level=logging.INFO, screen=False, to_file=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if to_file:
        log_file = os.path.join(save_dir, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def rgb2ycbcr(img_np):
    h, w, _ = img_np.shape
    y_map = np.zeros((h, w)).astype(np.float32)
    Y = 0.257*img_np[:,:, 2]+0.504*img_np[:,:, 1]+0.098*img_np[:,:, 0]+16

    return Y

def cal_dif(y_np_list):
    length = len(y_np_list)
    y_diff_list = []
    max_value, min_value = -100000, 100000
    for i in range(length-1):
        y_diff_list.append(np.abs(y_np_list[i+1]-y_np_list[i]))
        max_value = max(max_value, y_diff_list[i].max())
        min_value = min(min_value, y_diff_list[i].min())
    
    for i in range(length-1):
        y_diff_list[i] = (y_diff_list[i]-min_value)/max_value*255

    return y_diff_list


# 作为metric的SSIM
def gaussian(window_size, sigma):
    gauss = mge.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = F.expand_dims(gaussian(window_size, 1.5), 1)
    _2D_window = F.expand_dims(F.matmul(_1D_window, _1D_window.T), [0, 1])
    window = F.broadcast_to(_2D_window, (channel, 1, 1, window_size, window_size))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = F.pow(mu1, 2)
    mu2_sq = F.pow(mu2, 2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# 照着https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/init.py实现的
# 测试了下，结果是对上的
def compute_SSIM(warp_imgs, match_imgs):
    _, channel, _, _ = warp_imgs.shape
    window = create_window(11, channel)
    ssim = _ssim(warp_imgs, match_imgs, window, 11, channel, True)
    return ssim