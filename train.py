import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import torch.nn.functional as F

from collections import OrderedDict
import json
from PIL import Image
import argparse
import functions

parser = argparse.ArgumentParser()

 
parser.add_argument('--arch', type = str, default = 'vgg', 
                    help = 'choose CNN model architecture')
parser.add_argument('--lr', type = float, default = 0.001, help = 'Specify the learning rate')
parser.add_argument('--hu', type = int, default = 4096, help = 'Specify number of hidden units')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'Specify p of the dropout layer')
parser.add_argument('--epochs', type = int, default = 15, help = 'Specify number of epochs')
parser.add_argument('--train_batch_size', type = int, default = 64, help = 'Specify training dataset batch size')
parser.add_argument('--valid_batch_size', type = int, default = 32, help = 'Specify validation dataset batch size')
parser.add_argument('--test_batch_size', type = int, default = 32, help = 'Specify test dataset batch size')
parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'train using GPU or CPU, True=GPU, False=CPU')
parser.add_argument('--cp_path', type = str, default = 'checkpoint.pth', help = 'Path to the saved checkpoint')

in_args = parser.parse_args()

# Loading dataset
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms, validation_transforms,  test_transforms = functions.data_transforms(train_dir, valid_dir, test_dir)

train_dataset, validation_dataset, test_dataset = functions.load_dataset(train_dir, train_transforms, valid_dir, validation_transforms, test_dir, test_transforms)

train_loader, validation_loader, test_loader = functions.data_loaders(train_dataset, validation_dataset, test_dataset, in_args.train_batch_size, in_args.valid_batch_size, in_args.test_batch_size)

# Build the Neural Network
if in_args.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
elif in_args.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)
elif in_args.arch == 'resnet':
    input_size = 2048
    model = models.resnet50(pretrained=True)
print(model)

# Freeze  the pretrained model parameters
for parameter in model.parameters():
    parameter.requires_grad = False

# Build the classifier
classifier = nn.Sequential(OrderedDict([('inputs', nn.Linear(input_size, in_args.hu)),
                                        ('relu1', nn.ReLU()),
                                        ('drop', nn.Dropout(p=in_args.dropout)),
                                        ('hidden_layer1', nn.Linear(in_args.hu, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.lr)

# Train the classifier 
functions.train_classifier(model, train_loader, validation_loader, optimizer, criterion, in_args.epochs,  in_args.gpu)
    
accuracy = functions.testing(model, test_loader, in_args.gpu)
print("Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

functions.save_checkpoint(in_args.cp_path, model, train_dataset, optimizer, in_args.arch, in_args.epochs, input_size, in_args.hu, in_args.lr, in_args.dropout)  

    

