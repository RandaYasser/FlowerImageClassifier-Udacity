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

parser.add_argument('--img_path', type = str, default = 'flowers/test/1/image_06743.jpg', 
                    help = 'path to the image we want to predict')
parser.add_argument('--topk', type = int, default = 5, help = 'Show top k classes')
parser.add_argument('--json', type = str, default = 'cat_to_name.json', 
                    help = 'path to the json file of flowers categories')
parser.add_argument('--cp_path', type = str, default = 'checkpoint.pth', help = 'Path to the saved checkpoint')
parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'train using GPU or CPU, True=GPU, False=CPU')
in_args = parser.parse_args()

model = functions.load_checkpoint(in_args.cp_path)

cat_to_class = functions.load_json(in_args.json)

image = functions.process_image(in_args.img_path)

probabilities, classes = functions.predict(in_args.img_path, model, in_args.topk, in_args.gpu)
names = [cat_to_class[class_] for class_ in classes]
print(names)
print(probabilities)