import megengine as mge
import megengine 
from megengine import module as M
import megengine.functional as F
device = megengine.get_default_device()

def gauss_kernel(size=5, channels=3):
    kernel = mge.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = F.tile(kernel, (channels, 1, 1, 1))
    kernel = kernel
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = F.concat([x, F.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))], axis=3)
    cc = cc.reshape(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.transpose(0,1,3,2)
    cc = F.concat([cc, F.zeros((x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2))], axis=3)
    cc = cc.reshape(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.transpose(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = F.nn.pad(img, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='reflect')
    kernel = F.expand_dims(kernel, 1)
    out = F.nn.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(M.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(F.nn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))