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
    
class tdBN(nn.Module):
    def __init__(self, num_features, Vth=0.5, alpha=2.0, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.Vth = Vth
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            xf = x.float()
            mean = xf.mean(dim=(0, 2, 3))
            var = xf.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var
        mean = mean.to(x.dtype)[None, :, None, None]
        var = var.to(x.dtype)[None, :, None, None]
        x = self.alpha * self.Vth * (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class multispike_ste(torch.autograd.Function):
    # ilif
    @staticmethod
    def forward(ctx, x, levels):
        ctx.save_for_backward(x)
        ctx.levels = levels
        return torch.round(torch.clamp(x, 0.0, float(levels)))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        levels = ctx.levels
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        grad_input[x > levels] = 0
        return grad_input, None


class lif_fused(torch.autograd.Function):
    # memory-efficient
    # 不保存时间展开过程中的中间状态，反向传播时重新计算
    @staticmethod
    def forward(ctx, x, thresh, tau, gamma):
        T = x.shape[0]
        mem = torch.zeros_like(x[0])
        spikes = torch.empty_like(x)
        for t in range(T):
            mem = mem * tau + x[t]
            spike = (mem >= thresh).to(x.dtype)
            spikes[t] = spike
            mem = mem * (1.0 - spike)
        ctx.save_for_backward(x)
        ctx.thresh, ctx.tau, ctx.gamma = thresh, tau, gamma
        return spikes

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        thresh, tau, gamma = ctx.thresh, ctx.tau, ctx.gamma
        T = x.shape[0]
        us, ss = [], []
        mem = torch.zeros_like(x[0])
        for t in range(T):
            u = mem * tau + x[t]
            s = (u >= thresh).to(x.dtype)
            us.append(u)
            ss.append(s)
            mem = u * (1.0 - s)
        grad_x = torch.empty_like(x)
        grad_r = torch.zeros_like(x[0])
        inv_g2 = (1.0 / gamma) * (1.0 / gamma)
        for t in range(T - 1, -1, -1):
            u, s = us[t], ss[t]
            sg = inv_g2 * (gamma - (u - thresh).abs()).clamp(min=0)
            grad_u = grad_out[t] * sg + grad_r * ((1.0 - s) - u * sg)
            grad_x[t] = grad_u
            grad_r = tau * grad_u
        return grad_x, None, None, None


class LIF(nn.Module):
    
    record_rate = False
   
    mem_efficient = True

    def __init__(self, T, thresh=0.5, tau=0.5, gamma=2.0):
        super().__init__()
        self.act = lifact.apply
        self.T = T
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.totime = ExpandTemporalDim(self.T)
        self.tobatch = MergeTemporalDim(self.T)
        self.last_rate = None

    def forward(self, x):
        x = self.totime(x)
        if self.mem_efficient:
            spikes = lif_fused.apply(x, self.thresh, self.tau, self.gamma)
        else:
            mem = torch.zeros_like(x[0])
            buf = []
            for t in range(self.T):
                mem = mem * self.tau + x[t, ...]
                spike = self.act(mem - self.thresh, self.gamma)
                mem = (1 - spike) * mem
                buf.append(spike)
            spikes = torch.stack(buf, dim=0)
        if self.record_rate:
            self.last_rate = spikes.mean().detach()
        return self.tobatch(spikes)


def firing_rate_report(model):
    """Return {module_name: mean firing rate} for every LIF that has recorded one.
    Requires LIF.record_rate = True during the forward pass being inspected."""
    rates = {}
    for name, m in model.named_modules():
        if isinstance(m, LIF) and getattr(m, 'last_rate', None) is not None:
            rates[name] = m.last_rate.item()
    return rates