from numpy import dtype
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


""" jacc = torchmetrics.JaccardIndex(13)

y_pred = torch.randint(0,1,(32,13,256,256)) 
y = torch.randint(0,1 ,(32,13,256,256))

metrics = jacc(y,y_pred)
print(metrics) """
"""
y = torch.randn((32,13,256,256))
y = y.view(32,256,256,13)
print(y[0][0][0])
"""

""" jacc = torchmetrics.JaccardIndex(13)

y_pred = torch.randn((32,13,256,256)) 
y = torch.randint(0,1,(32,13,256,256))

metrics = jacc(y_pred, y)
print(metrics)  """

a = torch.Tensor([[1], [1.3]])
print(a[0].dtype)
