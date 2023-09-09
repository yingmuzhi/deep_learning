'''
The most importance about these two files are:
1. in model times, using PyTorch built-in optim, nn, loss
2. add L2 weight decay
'''
import torch
from core import *
from torch import nn


"""
1. Data
"""
class SyntheticRegressionData(DataModule):
    def __init__(self, 
                 w, 
                 b, 
                 noise=0.01,
                 num_train=1000,
                 num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise


# test
# data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# print('features shape:', data.X.shape,'\nlabel shape:', data.y.shape)
# print('features:', data.X[0],'\nlabel:', data.y[0])

@add_to_class(DataModule) 
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)    # ([x], [y])
    # dataset = torch.utils.data.dataset.TensorDataset(*tensors)
    dataset = MyDataset(*tensors)
    return torch.utils.data.dataloader.DataLoader(dataset, self.batch_size,
                                       shuffle=train)

class MyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)

# test
# X, y = next(iter(data.train_dataloader()))
# print('X shape:', X.shape, '\ny shape:', y.shape)
# print(len(data.train_dataloader()))


"""
2. Model
"""
class LinearRegression(ModelModule):
    """The linear regression model implemented from scratch."""
    def __init__(self, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        # init weight
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

# forward
@add_to_class(LinearRegression)
def forward(self, X):
    return self.net(X)

# loss
@add_to_class(LinearRegression)
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)

# algorithm
@add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr) 


"""
3. Trainer
"""
@add_to_class(TrainerModule) 
def prepare_batch(self, batch):
    return batch

@add_to_class(TrainerModule)
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return 
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            loss = self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1


model = LinearRegression(lr=0.03)
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = TrainerModule(max_epochs=3)
trainer.fit(model, data)
pass


"""
4. L2 loss
"""
class Data(DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)

def l2_penalty(w):
    return (w ** 2).sum() / 2

class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)

@add_to_class(WeightDecay)
def get_w_b(self):
    """Defined in :numref:`sec_linear_concise`"""
    return (self.net.weight.data, self.net.bias.data)

data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = TrainerModule(max_epochs=10)

model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))