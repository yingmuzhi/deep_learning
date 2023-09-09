import torch
from core import *


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
class LinearRegressionScratch(ModelModule):
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

# forward
@add_to_class(LinearRegressionScratch)
def forward(self, X):
    return torch.matmul(X, self.w) + self.b

# loss
@add_to_class(LinearRegressionScratch)
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean()

# algorithm
class SGD(HyperParameters):
    def __init__(self, params, lr) -> None:
        super().__init__()
        self.save_hyperparameters()
    
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
@add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr) 


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


model = LinearRegressionScratch(2, lr=0.03)
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = TrainerModule(max_epochs=3)
trainer.fit(model, data)
pass