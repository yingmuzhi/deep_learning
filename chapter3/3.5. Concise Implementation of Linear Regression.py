import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


"""
1. define model

"""
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1) # The latter allows users to only specify the output dimension | Specifying input shapes is inconvenient
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

# 使用bulit-in func `__call__` 实现forward
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)

"""
2. define loss

"""
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)

"""
3. define optimizer

"""
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)

"""
4. training

"""
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)

@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')