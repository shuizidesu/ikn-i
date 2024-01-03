import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base_model import KoopmanNet
from torch import nn, Tensor


def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n].contiguous()
    x2 = x[:, n:].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    """
    Expansion of the input dimension by pad zero

    Input dim: [B, D] = batch size * feature dimension

    Args:
        pad_size (int): The number of pad zero
    """
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size), 'constant', 0)
        return x

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size]


class imlp_block(nn.Module):
    """
    Args:
        in_ch (int): The dimension of input features, total
        hidden_ch (int): The dimension of hidden dimension
        out_ch (int): The dimension of output features, local (total = local * 2)
    """
    def __init__(self, in_ch, out_ch, hidden_ch):
        '''build invertible MLP bottleneck block'''
        super(imlp_block, self).__init__()
        self.pad_size = 2 * out_ch - in_ch
        self.inj_pad = injective_pad(self.pad_size)

        if self.pad_size !=0:
            in_ch = out_ch * 2

        layers = []
        layers.append(nn.Linear(in_ch//2, hidden_ch))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_ch, hidden_ch))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_ch, out_ch))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """bijective or injective block forward"""
        if self.pad_size != 0:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """bijective or injective block inverse"""
        x2, y1 = x[0], x[1]
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.pad_size != 0:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iMLPNet(nn.Module):
    """
    Build the iMLPNet

    Args:
        nBlocks (list): The number of imlp_block in the defined net (depth)
        nChannels (list): The output dimension of the given block
        nHiddens (list): The hidden dimension of the given block
    """
    def __init__(self, nBlocks, nChannels, nHiddens, in_shape=None):
        super(iMLPNet, self).__init__()
        self.in_ch = in_shape
        self.nBlocks = nBlocks

        self.stack = self.imlp_stack(imlp_block, nBlocks, nChannels, nHiddens, in_ch=self.in_ch)

    def imlp_stack(self, _block, nBlocks, nChannels, nHiddens, in_ch):
        """Create stack of imlp blocks"""
        block_list = nn.ModuleList()
        hiddens = []
        channels = []
        for channel, depth, hidden in zip(nChannels, nBlocks, nHiddens):
            hiddens = hiddens + ([hidden]*depth)
            channels = channels + ([channel]*depth)
        for channel, hidden in zip(channels, hiddens):
            block_list.append(
                _block(
                    in_ch,
                    channel,
                    hidden,
                )
            )
            in_ch = channel * 2
        return block_list

    def forward(self, x):
        """imlpnet forward"""
        n = self.in_ch//2
        out = (x[:, :n], x[:, n:])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        return out_bij

    def inverse(self, out_bij):
        """imlpnet inverse"""
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0], out[1])
        x = out
        return x


class InvertKoopmanNet(KoopmanNet):
    def __init__(
            self,
            x_dim,
            x_blocks,
            x_channels,
            x_hiddens,
            u_dim,
            u_blocks,
            u_channels,
            u_hiddens,
    ):
        super(InvertKoopmanNet, self).__init__()
        self.x_dim = x_dim
        self.x_blocks = x_blocks
        self.x_channels = x_channels
        self.x_hiddens = x_hiddens

        self.u_dim = u_dim
        self.u_blocks = u_blocks
        self.u_channels = u_channels
        self.u_hiddens = u_hiddens

        self.x_encode_net = iMLPNet(
            nBlocks=self.x_blocks,
            nChannels=self.x_channels,
            nHiddens=self.x_hiddens,
            in_shape=self.x_dim
        )

        self.u_encode_net = iMLPNet(
            nBlocks=self.u_blocks,
            nChannels=self.u_channels,
            nHiddens=self.u_hiddens,
            in_shape=self.x_dim+self.u_dim
        )

        self.x_emb_dim = x_channels[-1]*2
        self.u_emb_dim = u_channels[-1]*2
        self.lA = nn.Linear(self.x_emb_dim, self.x_emb_dim, bias=False)
        self.lB = nn.Linear(self.u_emb_dim, self.x_emb_dim, bias=False)

    def x_encoder(self, x: Tensor):
        return self.x_encode_net(x)

    def u_encoder(self, x: Tensor, u: Tensor):
        x_u = torch.cat([x, u], dim=1)
        return self.u_encode_net(x_u)

    def koopman_operation(self, x_emb: Tensor, u_emb: Tensor):
        return self.lA(x_emb) + self.lB(u_emb)

    def x_decoder(self, x_emb: Tensor):
        return self.x_encode_net.inverse(x_emb)

    def u_decoder(self, u_emb: Tensor):
        x_u = self.u_encode_net.inverse(u_emb)
        return x_u[:, self.x_dim:]


if __name__ == '__main__':

    model = InvertKoopmanNet(
        x_dim=2,
        x_blocks=[2, 2],
        x_channels=[8, 16],
        x_hiddens=[64, 128],
        u_dim=1,
        u_blocks=[2, 2],
        u_channels=[8, 16],
        u_hiddens=[64, 128],
    )

    x = torch.rand(512, 2)
    u = torch.rand(512, 1)

    x_emb = model.x_encoder(x)
    x_recon = model.x_decoder(x_emb)

    u_emb = model.u_encoder(x, u)
    u_recon = model.u_decoder(u_emb)

    error = torch.nn.MSELoss()

    print(f"x_error: {error(x, x_recon)}, u_error: {error(u, u_recon)}")
