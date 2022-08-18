import torch
import torch.nn as nn


class UpSample(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.up = nn.Upsample(None, r, 'nearest')
        device = next(self.parameters()).device
        self.a = nn.Parameter(torch.ones((1,), device=device))

    def forward(self, x):
        return self.a * self.up(x)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=1.0):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.linear = nn.Linear(c, c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        x = p + self.linear(p)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        x = x.permute(1, 2, 0).reshape(b, c, h, w)
        return x


class CT(nn.Module):
    # Transformer And Conv Two-way information flow interaction
    def __init__(self, c_, num_heads):
        super().__init__()
        self.T = TransformerLayer(c_, num_heads)
        self.C = Bottleneck(c_, c_)

    def forward(self, x):
        q = self.T(x)
        k = self.C(x)
        x = torch.cat([q, k], 1)
        return x


class CTN(nn.Module):
    # Nx Transformer And Conv Two-way information flow interaction
    def __init__(self, c1, c2, n):
        super().__init__()
        c_ = int(c1 / 2)
        self.convs = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[CT(c_, int(c_ / 32)) for _ in range(n)])
        self.conve = Conv(int(c_ * 2), c2, 1, 1)

    def forward(self, x):
        x = self.convs(x)
        x = self.m(x)
        x = self.conve(x)
        return x


if __name__ == '__main__':
    im = torch.randn(1, 32, 64, 64)
    m = UpSample()
    y = m(im)
    print(y.shape)
