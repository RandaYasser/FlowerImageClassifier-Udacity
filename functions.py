import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import torch.nn.functional as F

from workspace_utils import active_session
from collections import OrderedDict
import json
from PIL import Image

def data_transforms(train_dir, valid_dir, test_dir):
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
    return train_transforms, validation_transforms, test_transforms


def load_dataset(train_dir, train_transforms, valid_dir, validation_transforms, test_dir, test_transforms):   
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms) 
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms) 
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_dataset, validation_dataset, test_dataset

def data_loaders(train_dataset, validation_dataset, test_dataset, train_batch_size, valid_batch_size, test_batch_size):
    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=valid_batch_size) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size) 

    return train_loader, validation_loader, test_loader

def load_json(json_file):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def validation(model, validation_loader_, criterion, gpu):
    # Function for the validation pass
    valid_loss = 0
    accuracy = 0
    
    for images, labels in iter(validation_loader_):
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        # probabilities of the output 
        probs = torch.exp(output)
        
        top_p, top_class = probs.topk(1, dim=1)
        # check if the max probability matches the correct label 
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)) # converts from ByteTensor to FloatTensor
    
    return valid_loss, accuracy

def train_classifier(model, train_loader_, validation_loader_, optimizer, criterion, n_epochs, gpu):
    '''Trian the classifier and Printing trianing loss, validation loss and validation accuracy as the      training runs.'''
    with active_session():
                

        epochs = n_epochs
        steps = 0
        if gpu:
            model.to('cuda')
        print_every = 32
        print("Training started")
        for e in range(epochs):
        
            model.train()
    
            running_loss = 0
    
            for images, labels in iter(train_loader_):
        
                steps += 1
                if gpu:
                    images, labels = images.to('cuda'), labels.to('cuda')
        
                optimizer.zero_grad()
        
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                
                if steps % print_every == 0:    
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        model.eval()
                        validation_loss, accuracy = validation(model, validation_loader_, criterion, gpu)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader_)),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_loader_)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader_)))

                    running_loss = 0
                    model.train()
        print("Training finished, Your model is ready!")        
 
def testing(model, test_loader_, gpu):
    
    model.eval()
    if gpu:
        model.to('cuda')

    with torch.no_grad():
    
        test_accuracy = 0
    
        for images, labels in iter(test_loader_):
            
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            output = model.forward(images)
            probs = torch.exp(output)

            top_p, top_class = probs.topk(1, dim=1)
            # check if the max probability matches the correct label 
            equality = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equality.type(torch.FloatTensor)) # converts from ByteTensor to FloatTensor
            
    return test_accuracy 

def save_checkpoint(cp_path, model, train_dataset, optimizer, arch, n_epochs, inputs, hidden_units, lr, dropout):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
                  'arch': arch,
                  'epochs': n_epochs,
                  'inputs': inputs,
                  'hidden_units': hidden_units,
                  'dropout': dropout,
                  'learning_rate': lr,
                  'optimizer':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, cp_path)
    print("Checkpoint Saved path:{}".format(cp_path))
    


def load_checkpoint(cp_path):
    # Function that loads a checkpoint and rebuilds the model
    checkpoint = torch.load(cp_path)
    
    if checkpoint['arch'] == 'vgg':
        input_size = 25088
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        input_size = 9216
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'resnet':
        input_size = 2048
        model = models.resnet50(pretrained=True)    
    else:
        print("Arch used is {}".format(checkpoint['arch']))
    for param in model.parameters():
        param.requires_grad = False
            
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([('inputs', nn.Linear(checkpoint['inputs'], checkpoint['hidden_units'])),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=checkpoint['dropout'])),
                                            ('hidden_layer1', nn.Linear(checkpoint['hidden_units'], 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded Checkpoint Successfully")
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    image_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    return image_transforms(pil_image)
    

def predict(image_path, model, k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    if gpu:
        model.cuda()
        image = image.cuda()
        
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    probs = torch.exp(output)
    
    top_p, top_class = probs.topk(k, dim=1)
    probabilities = top_p.squeeze().tolist()
    indices = top_class.squeeze().tolist()
    
    # match top indices to the class label
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[index] for index in indices]
    
    return probabilities, classes
