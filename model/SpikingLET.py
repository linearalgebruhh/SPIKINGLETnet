import torch
import torch.nn as nn
import torch.nn.functional as F
from .SNNNeurons.transformer import TransBlock
from .SNNNeurons.patch import reverse_patches
from .SNNNeurons import neurons as bptt_neurons
from .SNNNeurons import neurons_ottt as ottt_neurons
from .SNNNeurons.neurons import tdBN

__all__ = ["LETNet"]

# Use threshold-dependent BatchNorm before every spiking neuron. Set False to
# fall back to vanilla BatchNorm2d (for A/B comparison).
USE_TDBN = False

DEFAULT_BLOCKS = (3, 12, 12, 3, 3, 3)


def _repeat_dilation_pattern(length, pattern):
    if length <= 0:
        return []

    repeats = (length + len(pattern) - 1) // len(pattern)
    return (pattern * repeats)[:length]

CONFIG = {
    1: {
        "dim": 144,
        "num_heads": 4,
        "mlp_ratio": 2,
        "C": 16,
        "blocks": (3, 4, 12, 3, 3, 3),
    },
    2: {
        "dim": 288,
        "num_heads": 8,
        "mlp_ratio": 4,
        "C": 32,
        "blocks": (3, 10, 12, 3, 3, 3),
    },
    3: {
        "dim": 432,
        "num_heads": 8,
        "mlp_ratio": 4,
        "C": 48,
        "blocks": (3, 17, 12, 3, 3, 3),
        "attn_dim": 144,
        "mlp_hidden": 72,
    }
}

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, T, dilation=(1, 1), groups=1, bn_acti=False, bias=False,
                 thresh=0.5, tau=0.5, gamma=2.0, lif_cls=bptt_neurons.LIF):
        super().__init__()
        self.T = T
        self.bn_acti = bn_acti
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut, self.T, thresh=self.thresh, tau=self.tau, gamma=self.gamma, lif_cls=lif_cls)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn, T, thresh=0.5, tau=0.5, gamma=2.0, lif_cls=bptt_neurons.LIF):
        # nIn: num featrues
        super().__init__()
        self.T = T
        self.bn = tdBN(nIn, Vth=thresh) if USE_TDBN else nn.BatchNorm2d(nIn, eps=1e-3)
        # self.acti = nn.PReLU(nIn)
        self.acti = lif_cls(self.T, thresh=thresh, tau=tau, gamma=gamma)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, T, d=1, kSize=3, dkSize=3, thresh=0.5, tau=0.5, gamma=2.0, lif_cls=bptt_neurons.LIF):
        super().__init__()
        self.T = T
        self.bn_relu_1 = BNPReLU(nIn, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False, T=self.T,
                               thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.conv3x1 = Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True, T=self.T,
                            thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.conv1x3 = Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True, T=self.T,
                            thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True, T=self.T,
                             thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True, T=self.T,
                             thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.ca11 = eca_layer(nIn // 2)
        
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1),
                              groups=nIn // 2, bn_acti=True, T=self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d),
                              groups=nIn // 2, bn_acti=True, T=self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.ca22 = eca_layer(nIn // 2)

        self.bn_relu_2 = BNPReLU(nIn // 2, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False, T=self.T,
                            thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.shuffle = ShuffleBlock(nIn // 2)
        
    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)
        
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)
        
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        output = self.shuffle(output + input)

        return output

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    
class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut, T, thresh=0.5, tau=0.5, gamma=2.0, lif_cls=bptt_neurons.LIF):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.T = T
        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1, T=self.T,
                    thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)
        self.max_pool = nn.MaxPool2d(2, stride=2,ceil_mode=True)
        self.bn_prelu = BNPReLU(nOut, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output

class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput, T, thresh=0.5, tau=0.5, gamma=2.0, lif_cls=bptt_neurons.LIF):
        super().__init__()
        self.T = T
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = tdBN(noutput, Vth=thresh) if USE_TDBN else nn.BatchNorm2d(noutput, eps=1e-3)
        # self.relu = nn.ReLU6(inplace=True)
        self.act = lif_cls(self.T, thresh=thresh, tau=tau, gamma=gamma)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
        
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out    
        
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize, T, bn_acti=False, bias=False, thresh=0.5, tau=0.5, gamma=2.0,
                 lif_cls=bptt_neurons.LIF):
        super().__init__()
        self.T = T
        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=lif_cls)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
                 

class LETNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=12, block_3=12, block_4=3, block_5 = 3, block_6 = 3,
                 T=8, thresh=0.5, tau=0.5, gamma=2.0, cfg_n=1, neuron_mode='bptt'):
        super().__init__()
        if neuron_mode not in ('bptt', 'ottt'):
            raise ValueError(f"Unsupported neuron_mode '{neuron_mode}', use 'bptt' or 'ottt'.")
        self.neuron_mode = neuron_mode
        self.lif_cls = bptt_neurons.LIF if neuron_mode == 'bptt' else ottt_neurons.LIF
        # et config

        if cfg_n not in CONFIG:
            raise KeyError(f"Unsupported cfg_n {cfg_n}. Available configs: {sorted(CONFIG)}")

        cfg = CONFIG[cfg_n]
        self.dim = cfg["dim"]
        self.num_heads = cfg["num_heads"]
        self.mlp_ratio = cfg["mlp_ratio"]
        self.C = cfg["C"]

        requested_blocks = (block_1, block_2, block_3, block_4, block_5, block_6)
        if requested_blocks == DEFAULT_BLOCKS:
            block_1, block_2, block_3, block_4, block_5, block_6 = cfg.get("blocks", DEFAULT_BLOCKS)

        # SNN stuff
        self.T = T
        if self.neuron_mode == 'bptt':
            self.totime = bptt_neurons.ExpandTemporalDim(self.T)
            self.tobatch = bptt_neurons.MergeTemporalDim(self.T)
        else:
            self.totime = None
            self.tobatch = None


        # let stuff
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.PReLU(32),
        )
        
        self.bn_prelu_1 = BNPReLU(32, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)

        self.downsample_1 = DownSamplingBlock(32, 64, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)

        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i),
                                        DABModule(64, self.T, d=2, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls))
        self.bn_prelu_2 = BNPReLU(64, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)

        # DAB Block 2
        dilation_block_2 = _repeat_dilation_pattern(block_2, [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32])
        self.downsample_2 = DownSamplingBlock(64, 128, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, self.T, d=dilation_block_2[i], thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls))
        self.bn_prelu_3 = BNPReLU(128, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)

        # DAB Block 3
        #dilation_block_3 = [2, 5, 7, 9, 13, 17]
        dilation_block_3 = _repeat_dilation_pattern(block_3, [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32])
        self.downsample_3 = DownSamplingBlock(128, self.C, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(self.C, self.T, d=dilation_block_3[i], thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls))
        self.bn_prelu_4 = BNPReLU(self.C, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)


        self.transformer1 = TransBlock(
            self.T,
            dim=self.dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=cfg.get("qkv_bias", False),
            attn_dim=cfg.get("attn_dim", None),
            mlp_hidden=cfg.get("mlp_hidden", None),
            attn_drop=cfg.get("attn_drop", 0.0),
            thresh=thresh,
            tau=tau,
            gamma=gamma,
            lif_cls=self.lif_cls,
        )
        
        
#DECODER
        dilation_block_4 = [2] * block_4
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
            self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                DABModule(self.C, self.T, d=dilation_block_4[i], thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls))
        self.upsample_1 = UpsampleingBlock(self.C, 16, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        self.bn_prelu_5 = BNPReLU(16, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)

        dilation_block_5 = [2] * block_5
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        DABModule(16, self.T, d=dilation_block_5[i], thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls))
        self.upsample_2 = UpsampleingBlock(16, 16, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        self.bn_prelu_6 = BNPReLU(16, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        
        
        dilation_block_6 = [2] * block_6
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        DABModule(16, self.T, d=dilation_block_6[i], thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls))
        self.upsample_3 = UpsampleingBlock(16, 16, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        self.bn_prelu_7 = BNPReLU(16, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        
        
        self.PA1 = PA(16)
        self.PA2 = PA(16)
        self.PA3 = PA(16)


        
        self.LC1 = LongConnection(64, 16, 3, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        self.LC2 = LongConnection(128, 16, 3, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        self.LC3 = LongConnection(self.C, self.C, 3, self.T, thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls)
        # LC0: init_conv /8 feature → decoder last stage (already in memory, zero extra cost)
        self.LC0 = nn.Conv2d(32, 16, kernel_size=1, bias=False)

        self.classifier = nn.Sequential(Conv(16, classes, 1, 1, padding=0, T=self.T,
                                             thresh=thresh, tau=tau, gamma=gamma, lif_cls=self.lif_cls))

    def reset_state(self):
        for module in self.modules():
            if module is self:
                continue
            reset_fn = getattr(module, 'reset_state', None)
            if callable(reset_fn):
                reset_fn()

    def _forward_core(self, x0, out_size):
        # x0: ANN-stem features already expanded into the (T*B) spiking domain.
        output0 = self.bn_prelu_1(x0)

        # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        output1 = self.bn_prelu_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        output2 = self.bn_prelu_3(output2)

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        output3 = self.bn_prelu_4(output3)

#Transformer

        b, c, h, w = output3.shape
        output4 = self.transformer1(output3)

        output4 = output4.permute(0, 2, 1)
        output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)

#DECODER
        output4 = self.DAB_Block_4(output4)
        lc3 = self.LC3(output3)
        if output4.shape[-2:] != lc3.shape[-2:]:
            lc3 = F.interpolate(lc3, size=output4.shape[-2:], mode='bilinear', align_corners=False)
        output4 = self.upsample_1(output4 + lc3)
        output4 = self.PA1(output4)
        output4 = self.bn_prelu_5(output4)
        
        
        output5 = self.DAB_Block_5(output4)
        lc2 = self.LC2(output2)
        if output5.shape[-2:] != lc2.shape[-2:]:
            lc2 = F.interpolate(lc2, size=output5.shape[-2:], mode='bilinear', align_corners=False)
        output5 = self.upsample_2(output5 + lc2)
        output5 = self.PA2(output5)
        output5 = self.bn_prelu_6(output5)
        
        
        output6 = self.DAB_Block_6(output5)
        lc1 = self.LC1(output1)
        if output6.shape[-2:] != lc1.shape[-2:]:
            lc1 = F.interpolate(lc1, size=output6.shape[-2:], mode='bilinear', align_corners=False)
        output6 = self.upsample_3(output6 + lc1)
        output6 = self.PA3(output6)
        output6 = self.bn_prelu_7(output6)

        # LC0: fuse init_conv /8 high-resolution features (already resident in memory)
        lc0 = self.LC0(output0)
        if output6.shape[-2:] != lc0.shape[-2:]:
            lc0 = F.interpolate(lc0, size=output6.shape[-2:], mode='bilinear', align_corners=False)
        output6 = output6 + lc0

        out = F.interpolate(output6, out_size, mode='bilinear', align_corners=False)
        out = self.classifier(out)
        return out

    def forward(self, input, **kwargs):
        out_size = input.size()[2:]
        # ANN stem runs ONCE on (B,3,H,W) -- real-valued, no time dimension.
        x0 = self.init_conv(input)

        if self.neuron_mode == 'bptt':
            x0 = x0.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            x0 = self.tobatch(x0)  # (T,B,...) -> (T*B,...): enter spiking domain
            out = self._forward_core(x0, out_size)
            out = self.totime(out).mean(0)
            return out

        init = kwargs.get('init', False)
        if init:
            self.reset_state()
        return self._forward_core(x0, out_size)


"""print layers and params of network"""
if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LETNet(classes=19).to(device)
    summary(model, (3, 512, 1024))
