#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal



print("🔍 Torch version:", torch.__version__)
print("🧠 CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using device:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ Running on CPU — no GPU found")

# Test: kreiraj tensor i prebaci na GPU
x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)
z = x + y

print("📦 x:", x)
print("📦 y:", y)
print("🧮 x + y:", z)



print("🔍 Checking environment...\n")

# Basic numerical operations
array = np.array([1, 2, 3])
print("✅ NumPy array sum:", np.sum(array))

# Math rounding
value = 3.678
print("✅ Rounded floor:", floor(value), "Rounded ceil:", ceil(value))

# Torch: GPU availability
cuda_available = torch.cuda.is_available()
print("🧠 CUDA available:", cuda_available)
if cuda_available:
    print("🚀 GPU name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ No GPU detected — running on CPU.")

# Torch tensor + NN forward pass
x = torch.rand(5, 10)
linear = nn.Linear(10, 3)
output = linear(x)
print("✅ Torch Linear output shape:", output.shape)

# Distributions test
dist_cat = Categorical(logits=torch.ones(5))
sample_cat = dist_cat.sample()
print("🎲 Categorical sample:", sample_cat.item())

dist_norm = Normal(torch.tensor(0.0), torch.tensor(1.0))
sample_norm = dist_norm.sample()
print("🎲 Normal sample:", sample_norm.item())

# Matplotlib test
plt.plot([1, 2, 3], [3, 2, 1], color='skyblue')
plt.title("🖼️ Matplotlib Test Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.savefig("matplotlib_test.png")
print("✅ Matplotlib plot saved as 'matplotlib_test.png'.")

print("\n✅ Environment test complete — all core packages operational!")


