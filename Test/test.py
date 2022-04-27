import time
import torch.nn as nn
import torchvision.models
from tqdm import tqdm
import torch

# # Image Example
# N, C, H, W = 20, 5, 10, 10
# input = torch.randn(N, C, H, W)
# # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# # as shown in the image below
# layer_norm = nn.LayerNorm([C, H, W])
# output = layer_norm(input)
# print(output)

net = torchvision.models.resnet18()
print(net)