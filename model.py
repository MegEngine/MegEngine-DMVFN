import numpy as np
import itertools
from warplayer import warp
import megengine as mge
import megengine.optimizer as optim
import megengine.distributed as dist
import megengine.autodiff as ad

from arch import *
from loss import *
from laplacian import *

device = megengine.get_default_device()
    
class Model:
    def __init__(self, local_rank=-1, resume_path=None, resume_epoch=0, load_path=None, simply_infer=False):
        self.dmvfn = DMVFN()
        self.optimG = optim.AdamW(self.dmvfn.parameters(), lr=1e-6, weight_decay=1e-3)
        self.lap = LapLoss()

        if simply_infer == False:
            dist.bcast_list_(self.dmvfn.tensors())
            self.gm = ad.GradManager()
            self.gm.attach(self.dmvfn.parameters(), callbacks=[dist.make_allreduce_cb("mean")])

        if resume_path is not None:
            assert resume_epoch>=1
            print(local_rank,": loading...... ", '{}/dmvfn_{}.pkl'.format(resume_path, str(resume_epoch-1)))
            self.dmvfn.load_state_dict(mge.load('{}/dmvfn_{}.pkl'.format(resume_path, str(resume_epoch-1))), strict=True)
        else:
            if load_path is not None:
                self.dmvfn.load_state_dict(mge.load(load_path), strict=True)

    def train(self, imgs, learning_rate=0):
        self.dmvfn.train()
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        b, n, c, h, w = imgs.shape
        loss_avg = 0
        for i in range(n-2):
            with self.gm:
                img0, img1, gt = imgs[:, i], imgs[:, i+1], imgs[:, i+2]
            
                merged =  self.dmvfn(F.concat([img0, img1, gt], 1), scale=[4,4,4,2,2,2,1,1,1])
                loss_G, loss_l1= 0, 0
                for i in range(9):
                    loss_l1 +=  (self.lap(merged[i], gt)).mean()*(0.8**(8-i))

                loss_G = loss_l1
                self.gm.backward(loss_G)
            optim.clip_grad_norm(self.dmvfn.parameters(), 1.0)
            self.optimG.step().clear_grad()
            loss_avg += loss_G

        return loss_avg/(n-2)

    def eval(self, imgs, name='city', scale_list = [4,4,4,2,2,2,1,1,1]):
        self.dmvfn.eval()
        b, n, c, h, w = imgs.shape 
        preds = []
        if name == 'CityValDataset':
            assert n == 14
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for i in range(5):
                merged= self.dmvfn(F.concat((img0, img1), 1), scale=scale_list, training=False)
                length = len(merged)
                if length == 0:
                    pred = img0
                else:
                    pred = merged[-1]

                preds.append(pred)
                img0 = img1
                img1 = pred
            assert len(preds) == 5
        elif name == 'KittiValDataset' or name == 'DavisValDataset':
            assert n == 9
            img0, img1 = imgs[:, 2], imgs[:, 3]
            for i in range(5):
                merged = self.dmvfn(F.concat((img0, img1), 1), scale=scale_list, training=False)
                length = len(merged)
                if length == 0:
                    pred = img0
                else:
                    pred = merged[-1]
                preds.append(pred)
                img0 = img1
                img1 = pred
            assert len(preds) == 5
        elif name == 'VimeoValDataset':
            assert n == 3
            merged = self.dmvfn(F.concat((imgs[:, 0], imgs[:, 1]), 1), scale=scale_list, training=False)
            length = len(merged)
            if length == 0:
                pred = imgs[:, 0]
            else:
                pred = merged[-1]
            preds.append(pred)
            assert len(preds) == 1
        return F.stack(preds, 1)

    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            mge.save(self.dmvfn.state_dict(),'{}/dmvfn_{}.pkl'.format(path, str(epoch)))
