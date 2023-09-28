import torch
import core


# region activation functions
"""

1. relu
"""
# relu sub
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True).reshape(2, 80)
y = torch.relu(x)
core.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))     # detach 脱离计算图

# relu orgin:: means [torch.Tensor]
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
core.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))     # detach 脱离计算图

# relu grad
y.backward(torch.ones_like(x), retain_graph=True)
core.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))


"""

2. sigmoid
"""
y = torch.sigmoid(x)
core.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# grad
x.grad.data.zero_() # like optim.zero_grad()    # 这里要想明白的是针对x的梯度清零，不是y的，对谁求偏导则是对谁的梯度清零
y.backward(torch.ones_like(x), retain_graph=True)
core.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))


"""

3. Tanh
"""
y = torch.tanh(x)
core.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

# grad
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
core.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
# endregion


# region building MLP
"""

1. model
"""
# model and parameters init
class MLPScratch(core.Classifier):
    def __init__(self, 
                 num_inputs, 
                 num_outputs,
                 num_hiddens, 
                 lr, 
                 sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = torch.nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = torch.nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = torch.nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = torch.nn.Parameter(torch.zeros(num_outputs))

# activation
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

@core.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)    # since X is (sample, features), such X is on the left
    return torch.matmul(H, self.W2) + self.b2

# train
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = core.FashionMNIST(batch_size=256, download=False)
trainer = core.TrainerModule(max_epochs=10)
trainer.fit(model, data)
# endregion