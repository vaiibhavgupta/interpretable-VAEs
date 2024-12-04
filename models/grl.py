import torch

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return - grad_output * ctx.lambda_, None