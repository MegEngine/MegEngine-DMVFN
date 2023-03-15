import megengine.functional as F
import megengine
import megengine.module as M
from megengine.autodiff import Function
import megengine.random as rand
# from scipy.stats import bernoulli
import numpy as np

device = megengine.get_default_device()

backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size))
    if k not in backwarp_tenGrid:
        tenHorizontal = F.broadcast_to(F.linspace(0, 1.0, tenFlow.shape[3]).reshape(
            1, 1, 1, tenFlow.shape[3]),(tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3])) * (tenFlow.shape[3] - 1)
        tenVertical = F.broadcast_to(F.linspace(0, 1.0, tenFlow.shape[2]).reshape(
            1, 1, tenFlow.shape[2], 1),(tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3])) * (tenFlow.shape[2] - 1)
        backwarp_tenGrid[k] = F.concat(
            [tenHorizontal, tenVertical], 1)

    g = (backwarp_tenGrid[k] + tenFlow).transpose(0, 2, 3, 1)
    return F.nn.remap(inp=tenInput, map_xy=g)

class RoundSTE(Function):
    def forward(self, x):
        b, c = x.shape
        y = rand.uniform(size=(b, c))

        one_hot = F.zeros_like(y)
        one_hot[y<x] = 1
        return one_hot

    def backward(self, grad):
        return grad, None


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return M.Sequential(
        M.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        M.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return M.Sequential(
        M.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        M.PReLU(out_planes)
    )


class MVFB(M.Module):
    def __init__(self, in_planes, num_feature):
        super(MVFB, self).__init__()
        self.conv0 = M.Sequential(
            conv(in_planes, num_feature//2, 3, 2, 1),
            conv(num_feature//2, num_feature, 3, 2, 1),
            )
        self.convblock = M.Sequential(
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
        )
        self.conv_sq = conv(num_feature, num_feature//4)

        self.conv1 = M.Sequential(
            conv(in_planes, 8, 3, 2, 1),
        )
        self.convblock1 = M.Sequential(
            conv(8, 8),
        )
        self.lastconv = M.ConvTranspose2d(num_feature//4 + 8, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        x0 = x
        flow0 = flow
        if scale != 1:
            x = F.nn.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        flow = F.nn.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = F.concat((x, flow), 1)
        x1 = self.conv0(x)
        x2 = self.conv_sq(self.convblock(x1) + x1)
        x2 = F.nn.interpolate(x2, scale_factor=scale * 2, mode="bilinear", align_corners=False)

        x3 = self.conv1(F.concat((x0,flow0), 1))
        x4 = self.convblock1(x3)
        tmp = self.lastconv(F.concat((x2, x4), axis=1))
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        return flow, mask



class DMVFN(M.Module):
    def __init__(self):
        super(DMVFN, self).__init__()
        self.block0 = MVFB(13+4, num_feature=160)
        self.block1 = MVFB(13+4, num_feature=160)
        self.block2 = MVFB(13+4, num_feature=160)
        self.block3 = MVFB(13+4, num_feature=80)
        self.block4 = MVFB(13+4, num_feature=80)
        self.block5 = MVFB(13+4, num_feature=80)
        self.block6 = MVFB(13+4, num_feature=44)
        self.block7 = MVFB(13+4, num_feature=44)
        self.block8 = MVFB(13+4, num_feature=44)

        self.routing = M.Sequential(
            M.Conv2d(6, 32, 3, 1, 1),
            M.ReLU(),
            M.Conv2d(32, 32, 3, 1, 1),
            M.AdaptiveAvgPool2d((1, 1)),
        )
        self.l1 = M.Linear(32, 9)
        self.ste = RoundSTE()

    def forward(self, x, scale, training=True):
        batch_size, _, height, width = x.shape
        # print(x.shape)
        routing_vector = self.routing(x[:, :6]).reshape(batch_size, -1)
        routing_vector = F.sigmoid(self.l1(routing_vector))
        routing_vector = routing_vector / (routing_vector.sum(1, True) + 1e-6) * 4.5
        routing_vector = F.clip(routing_vector, 0, 1)
        ref = self.ste(routing_vector)

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        flow_list = []
        merged_final = []
        mask_final = []
        warped_img0 = img0
        warped_img1 = img1
        flow = F.zeros((batch_size, 4, height, width))
        mask = F.zeros((batch_size, 1, height, width))

        stu = [self.block0, self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7,
               self.block8]

        if training:
            for i in range(9):
                flow_d, mask_d = stu[i](F.concat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                        scale=scale[i])

                flow_right_now = flow + flow_d
                mask_right_now = mask + mask_d

                flow = flow + (flow_d) * ref[:, i].reshape(batch_size, 1, 1, 1)
                mask = mask + (mask_d) * ref[:, i].reshape(batch_size, 1, 1, 1)
                flow_list.append(flow)

                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])

                warped_img0_right_now = warp(img0, flow_right_now[:, :2])
                warped_img1_right_now = warp(img1, flow_right_now[:, 2:4])

                if i < 8:
                    mask_final.append(F.sigmoid(mask_right_now))
                    merged_student_right_now = (warped_img0_right_now, warped_img1_right_now)
                    merged_final.append(merged_student_right_now)
                else:
                    mask_final.append(F.sigmoid(mask))
                    merged_student = (warped_img0, warped_img1)
                    merged_final.append(merged_student)

            for i in range(9):
                merged_final[i] = merged_final[i][0] * mask_final[i] + merged_final[i][1] * (1 - mask_final[i])
                merged_final[i] = F.clip(merged_final[i], 0, 1)
            return merged_final
        else:
            for i in range(9):
                if ref[0, i]:
                    flow_d, mask_d = stu[i](F.concat((img0, img1, warped_img0, warped_img1, mask), 1), flow,
                                            scale=scale[i])
                    flow = flow + flow_d
                    mask = mask + mask_d

                    mask_final.append(F.sigmoid(mask))
                    flow_list.append(flow)
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    merged_student = (warped_img0, warped_img1)
                    merged_final.append(merged_student)
            length = len(merged_final)
            for i in range(length):
                merged_final[i] = merged_final[i][0] * mask_final[i] + merged_final[i][1] * (1 - mask_final[i])
                merged_final[i] = F.clip(merged_final[i], 0, 1)
            return merged_final


if __name__ == '__main__':
    net = DMVFN()
    x = megengine.tensor(np.random.rand(2, 6, 64, 64))
    y = net(x, scale=[4,4,4,2,2,2,1,1,1])
    print(y[-1].shape)
