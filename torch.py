def stable_linear(linear, operand):
    return torch.stack([
        linear(operand[...,idx,:]) for idx in range(operand.shape[-2])
    ], dim=-2)
def stable_exp(operand):
    return torch.stack([
        torch.exp(operand[...,idx]) for idx in range(operand.shape[-1])
    ], dim=-1)
def stable_conv1d(data, kernel, groups='ignored'):
    data_size = data.shape[-1]
    kernel_size = kernel.shape[-1]
    return torch.stack([
        (data[...,idx:idx+kernel_size-1] * kernel.transpose(-2,-3)[...,:-1]).sum(dim=-1) + data[...,idx+kernel_size-1] * kernel.transpose(-2,-3)[...,-1]
        for idx in range(0,data_size-kernel_size+1)
    ], dim=-1)

def tail_linear(linear, operand, ct=1):
    return torch.cat((
        linear(operand[:,:-ct,:]),
        linear(operand[:,-ct:,:])
    ), dim=-2)
def tail_exp(operand, ct=1):
    return torch.cat((
        torch.exp(operand[...,:-ct]),
        torch.exp(operand[...,-ct:])
    ), dim=-1)
def tail_sum(operand, ct=1):
    return operand[...,:-ct].sum(dim=-1) + operand[...,-ct:].sum(dim=-1)
def tail_conv1d(data, kernel, groups='ignored'):
    return torch.cat((
        torch.conv1d(data[...,:-1], kernel, groups=groups),
        tail_sum(
            data[...,-kernel.shape[-1]:] * kernel.transpose(-2,-3)
        )[...,None]
    ), dim=-1)
