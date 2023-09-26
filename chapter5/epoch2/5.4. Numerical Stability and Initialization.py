
"""

5.4.1.2. Exploding Gradients
"""
import torch
from d2l import torch as d2l


M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)