import torch


class GradNorm(torch.nn.Module):

    def __init__(self, moving_value=1e-4) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))
        self.register_full_backward_hook(self.backward)
        self.moving_value = moving_value

    def forward(x):
        return x
    
    @staticmethod
    def backward(module, grad_input, grad_output):
        mean, std = torch.std_mean(grad_output)
        module.scale = (1-module.moving_value) * module.scalar + module.moving_value * std
        module.bias = (1-module.moving_value) * module.bias + module.moving_value * mean
        return (module.scale * grad_output[0] + module.bias, )
