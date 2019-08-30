import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import torchvision
from torchvision import datasets, transforms, models

import PIL
from PIL import Image


import time 
import seaborn as sb

import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="deep_train")
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    parser.add_argument('--data_dir', action="store")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", choices=["vgg13", "vgg16"])
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default="0.001")
    parser.add_argument('--hidden_units', dest="hidden_units", action="store", default="4096")
    parser.add_argument('--epochs', dest="epochs", action="store", default="5")
    parser.add_argument('--gpu', action="store", default="gpu")
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, arg, classifier, epochs, train_data):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'arch': 'vgg16',
                  'classifier' : model.classifier,
                  'learning_rate': 0.001,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),}
    
    torch.save(checkpoint, path)
       

def train(model, criterion, optimizer, trainloader, validloader, testloader, epochs, gpu):
    steps = 0
    print_every = 50
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1 
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') 
            else:
                model.cpu()
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in testloader:
                        if gpu == 'gpu':
                            model.cuda()
                            inputs, labels = inputs.to('cuda'), labels.to('cuda') 
                        else:
                            model.cpu()
                            
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(testloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0

def main():
    print("start") 
    arg = arg_parse()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    
    model = getattr(models, arg.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if arg.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    elif arg.arch == "vgg16":
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(arg.learning_rate))
    epochs = int(arg.epochs)
    class_index = train_data.class_to_idx
    gpu = arg.gpu 
    train(model, criterion, optimizer, trainloader, validloader, testloader, epochs, gpu)
    model.class_to_idx = class_index
    path = arg.save_dir 
    save_checkpoint(path, model, optimizer, arg, classifier, epochs, train_data)


if __name__ == "__main__":
    main()
            
