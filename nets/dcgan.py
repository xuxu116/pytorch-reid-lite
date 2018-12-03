import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch import Tensor
from torch import nn
from torch.nn import Parameter


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    BATCH_SIZE = real_data.shape[0]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE,
            real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3,
                    256, 128)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, 
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    

# Generator Code
class Generator(nn.Module):
    def __init__(self, nz=256, nc=3, ngf=64, kernel_size=(8,4),
            imageNet_static=True):
        super(Generator, self).__init__()
        self.imageNet_static = imageNet_static
        
        if self.imageNet_static:
            mean = torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
            var = torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]])
            self.re_norm_mean = torch.nn.Parameter(mean, requires_grad = False)
            self.re_norm_var = torch.nn.Parameter(var, requires_grad = False)
            self.register_buffer('m_const', self.re_norm_mean)
            self.register_buffer('v_const', self.re_norm_var)
            
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            #nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1)
        if self.imageNet_static:
            output = (torch.nn.functional.sigmoid(self.main(input)) - self.m_const) / self.v_const
        else:
            output = (torch.nn.functional.tanh(self.main(input)))
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=32, kernel_size=(8,4), cls_id=False):
        super(Discriminator, self).__init__()
        # cls_id is to toggle re-id classification
        self.cls_id = cls_id
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf) x 32 x 32
            SpectralNorm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf) x 32 x 32
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)

        )
            # state size. (ndf*8) x 4 x 4
        self.cls_2 = nn.Conv2d(ndf * 8, 1, kernel_size, 1, 0, bias=False)
        if cls_id:
            self.fc = nn.Conv2d(ndf * 8, config["num_labels"], kernel_size, 1, 0, bias=True)
        

    def forward(self, input):
        x = self.main(input)
        x_1 = self.cls_2(x)
        x_1 = x_1.view(-1)
        if self.cls_id:
            x_2 = self.fc(x)
            return x_1, x_2
        else:
            return x_1
    
