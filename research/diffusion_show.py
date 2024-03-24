import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
from tqdm import tqdm

class DiffusionPara():

    def __init__(self, start, end, steps, schedule='linear'):
        if schedule == 'linear':
            self.betas = torch.linspace(start, end, steps)
        elif schedule == "quad":
            self.betas = torch.linspace(start ** 0.5, end ** 0.5, steps) ** 2
        elif schedule == "sigmoid":
            self.betas = torch.linspace(-6, 6, steps)
            self.betas = torch.sigmoid(self.betas) * (end - start) + start
        else:
            raise Exception("wrong schedule.")

        self.alphas = 1 - self.betas
        self.alphas_sqrt = torch.sqrt(self.alphas)
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
    
    def get_para_of_x0(self):
        return self.alphas_prod * self.shift(self.betas) / (1 - self.shift(self.alphas_prod))

    def get_para_of_xt(self, t):
        return self.alphas_sqrt[t] * (1-self.alphas_prod[t-1]) / (1-self.alphas_prod[t])
    
    def get_alphas_bar(self, t):
        return self.alphas_prod[t]

    @staticmethod
    def shift(vec):
        return torch.cat([torch.tensor([10]).float(), vec[:-1]], 0)

def prepare(data: torch.Tensor, dp: DiffusionPara):
    return torch.mean(data.unsqueeze(-1) * dp.get_para_of_x0().unsqueeze(0).unsqueeze(0), dim=0)

def show(datas: torch.Tensor):
    d = datas.numpy()
    plt.scatter(d[:,0], d[:,1])
    plt.show()

def show_vec(t,dp, parts):
    w = 3
    h = 3
    splits = 31

    x = []
    y = []
    u = []
    v = []
    for i in np.linspace(-w, w, splits):
        for j in np.linspace(-h, h, splits):
            x.append(i)
            y.append(j)
            vec = dp.get_para_of_xt(t+1) * np.array([i, j]) + parts[t, :]
            u.append(vec[0])
            v.append(vec[1])
    plt.quiver(x,y,u,v)
    plt.show()

norm_dis = torch.distributions.normal.Normal(0, 1)

def do_reverse(dp: DiffusionPara, datas: torch.Tensor, xts: torch.Tensor, t):
    batch, dim = datas.size()
    bar_alpha = dp.get_alphas_bar(t)
    for_lookup = (xts.view([batch, 1, dim]) - torch.sqrt(bar_alpha)*datas.view([1, batch, dim]))/torch.sqrt(1-bar_alpha)
    # batch * batch
    prob = torch.exp(torch.sum(norm_dis.log_prob(torch.tensor(for_lookup)), dim=-1))

    # batch * batch * dim
    miu = dp.get_para_of_xt(t)*xts.view([batch, 1, dim]) + dp.get_para_of_x0()[t]*datas.view([1, batch, dim])

    # batch * dim
    upper = torch.mean(prob.unsqueeze(-1) * miu, dim=1)

    # batch
    downer = torch.mean(prob, dim=1)

    result_might_with_nan = upper / downer.view([batch, 1])

    return result_might_with_nan.masked_fill(torch.isnan(result_might_with_nan), value=0) + xts.masked_fill(~torch.isnan(result_might_with_nan), value=0)


num_steps = 100

swiss_roll, _ = make_swiss_roll(10**4,noise=0.1)
swiss_roll = swiss_roll[:, [0, 2]]/10.0

data = swiss_roll.T
dataset = torch.Tensor(data.T).float()
show(dataset)

dp = DiffusionPara(start=1e-5, end=0.5e-2, steps=100)
print(dp.get_para_of_x0())
print(dp.get_para_of_xt(1))

# parts = prepare(dataset, dp).permute([1,0])
# print(parts)
# show_vec(99, dp, parts)
xts = torch.randn(dataset.shape)
show(xts)
x_ = xts
for t in tqdm(reversed(range(num_steps-1))):
    t = t + 1
    x_ = do_reverse(dp, dataset, x_, t)
    if t==3:
        show(x_)
print(x_)
show(x_)