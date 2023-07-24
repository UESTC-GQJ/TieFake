from torchvision import models
from torch import nn
import numpy as np
import torch
import os
import sys
from resnest.torch import resnest101
def vgg19_2way(pretrained: bool):
    model_ft = models.vgg19(pretrained=pretrained)
    model_ft.classifier.add_module("fc",nn.Linear(1000,1))
    return model_ft

def resnet50_2way(pretrained: bool):
    model_ft = models.resnet50(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    return model_ft

def resnest101_2way(pretrained:bool):
    model_ft = resnest101(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    return model_ft

if __name__ == "__main__":
    model_ft = resnest101_2way(pretrained=True)
