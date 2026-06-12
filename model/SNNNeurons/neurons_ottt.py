import torch
import torch.nn as nn

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        xshape = [self.T, int(x_seq.shape[0]/self.T)]
        xshape.extend(x_seq.shape[1:])
        return x_seq.view(xshape)


class lifact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input >= 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, gamma0) = ctx.saved_tensors
        gamma = gamma0.item()
        grad_input = grad_output
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_output * tmp
        return grad_input, None
    
class LIF(nn.Module):
    def __init__(self, T, thresh=0.5, tau=2.0, gamma=2.0, decay_input=False, v_reset=None,
                 track_rate=True):
        super().__init__()
        self.act = lifact.apply
        self.T = T
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.track_rate = track_rate
        self.mem = None
        self.rate_tracking = None
        self.spike = None
        self.step = 0
        self.detach_interval = 1

    def reset_state(self):
        self.mem = None
        self.rate_tracking = None
        self.spike = None
        self.step = 0

    def forward_init(self, x):
        self.mem = torch.zeros_like(x)
        self.rate_tracking = None
        self.spike = None

    def forward(self, x, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        output_type = kwargs.get('output_type', 'spike')

        if init or self.mem is None or self.mem.shape != x.shape:
            self.forward_init(x)

        if self.step % self.detach_interval == 0:
            self.mem = self.mem.detach()

        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            self.mem = self.mem * (1 - 1. / self.tau) + x
        else:
            self.mem = self.mem * (1 - 1. / self.tau) + self.v_reset / self.tau + x
        spike = self.act(self.mem - self.thresh, self.gamma)
        self.mem = (1 - spike) * self.mem

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking is None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    self.rate_tracking = self.rate_tracking * (1 - 1. / self.tau) + spike.clone().detach()

        self.step += 1
        if output_type == 'spike_rate':
            assert self.track_rate, 'output_type="spike_rate" requires track_rate=True'
            return torch.cat((spike, self.rate_tracking), dim=0)
        return spike