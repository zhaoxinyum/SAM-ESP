import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPmodule(nn.Module):
    def __init__(
            self,
            iterations=50,
            entropy_epsilon=1.0,
            lam=1.0,
            ker_halfsize=2
    ):
        """
        :param iterations :iterations number
        """
        super(ESPmodule, self).__init__()
        self.iterations = iterations
        # Fixed paramaters of Gaussian function
        self.sigma = torch.full((1, 1, 1), 5.0, dtype=torch.float, requires_grad=False)
        self.ker_halfsize = ker_halfsize
        # Fixed paramater
        self.entropy_epsilon = entropy_epsilon
        self.lam = lam
        self.tau = 1 * self.entropy_epsilon

        self.nabla = nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]],
                                                [[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]], requires_grad=False))

        self.div = nn.Parameter(torch.tensor([[[[0., -1., 0.],
                                                [0., 1., 0.],
                                                [0., 0., 0.]],
                                               [[0., 0., 0.],
                                                [-1., 1., 0.],
                                                [0., 0., 0.]]]], requires_grad=False))

    def forward(self, o):
        # mask shape:(B,1,H,W),
        o = torch.squeeze(o, dim=1)
        # o shape:(B, H, W)

        u = torch.sigmoid(o / self.entropy_epsilon)

        # std kernel
        ker = self.STD_Kernel(self.sigma, self.ker_halfsize)
        ker = ker.to(o.device)
        # main iteration
        q = torch.zeros_like(o, device=o.device)
        x, y = torch.meshgrid(torch.arange(1, o.shape[1] + 1), torch.arange(1, o.shape[2] + 1))
        # x, y = torch.meshgrid(torch.arange(1, o.shape[1] + 1), torch.arange(1, o.shape[2] + 1))
        x = x.to(o.device)
        y = y.to(o.device)
        for i in range(self.iterations):
            # 1.regularization

            p = F.conv2d(1.0 - 2.0 * u.unsqueeze(1), ker, padding=self.ker_halfsize)
            # tangent direction

            Tx, Ty, centroid_x, centroid_y = self.tangent_direction(u, x, y)
            Tn = torch.sqrt(Tx ** 2 + Ty ** 2) + 1e-10
            Tx = Tx / Tn
            Ty = Ty / Tn

            # 2.shape dual variation
            u_nabla = F.conv2d(u.unsqueeze(1), weight=self.nabla, stride=1, padding=1)
            q = q - self.tau * (u_nabla[:, 0, :, :] * Tx + u_nabla[:, 1, :, :] * Ty)

            Tq = F.conv2d(torch.stack([Tx * q, Ty * q], dim=1), weight=self.div, padding=1)
            # Tq:(B,1,H,W)
            # 3.sigmoid
            u = torch.sigmoid((o - self.lam * p.squeeze(dim=1) - Tq.squeeze(dim=1)) / self.entropy_epsilon)

        u1 = (o - self.lam * p.squeeze(dim=1) - Tq.squeeze(dim=1)) / self.entropy_epsilon
        return u1

    def tangent_direction(self, u, x, y):
        # the shape of u:(B,H,W)
        centroid_x = torch.sum(u * x, dim=(1, 2)) / torch.sum(u, dim=(1, 2))
        centroid_y = torch.sum(u * y, dim=(1, 2)) / torch.sum(u, dim=(1, 2))
        centroid_x = centroid_x.view(-1, 1, 1)
        centroid_y = centroid_y.view(-1, 1, 1)
        # centroid_x/centroid_y:(B,1,1)
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        x, y = x.expand(u.shape[0], -1, -1), y.expand(u.shape[0], -1, -1)
        # x/y:(B,H,W)
        cossin = torch.sum((x - centroid_x) * (y - centroid_y) * u, dim=(1, 2)) / torch.sum(u, dim=(1, 2))
        bsin_acos = torch.sum(((x - centroid_x) ** 2) * u, dim=(1, 2)) / torch.sum(u, dim=(1, 2))
        asin_bcos = torch.sum(((y - centroid_y) ** 2) * u, dim=(1, 2)) / torch.sum(u, dim=(1, 2))
        # 椭圆切线方向
        sx = -(x - centroid_x) * cossin.view(-1, 1, 1) + (y - centroid_y) * bsin_acos.view(-1, 1, 1)
        sy = (y - centroid_y) * cossin.view(-1, 1, 1) - (x - centroid_x) * asin_bcos.view(-1, 1, 1)

        # 圆形约束
        # sx = y - centroid_x
        # sy = centroid_y - x
        return sx, sy, centroid_x, centroid_y

    def STD_Kernel(self, sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        # x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1),
        #                       indexing='ij')
        ker = torch.exp(-(x.float() ** 2 + y.float() ** 2) / (2.0 * sigma ** 2))
        ker = ker / (ker.sum(-1, keepdim=True).sum(-2, keepdim=True) + 1e-15)
        ker = ker.unsqueeze(1)
        return ker


