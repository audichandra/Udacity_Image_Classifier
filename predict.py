import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json

import torchvision
from torchvision import datasets, transforms, models

import PIL
from PIL import Image

import time 
import seaborn as sb

import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="deep_predict")
    parser.add_argument('--checkpoint', default="checkpoint.pth", action="store")
    parser.add_argument('--top_k', dest="top_k", default="3")
    parser.add_argument('--image_path', default='flowers/test/90/image_04405.jpg', action="store")
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', action="store", default="gpu")
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def load_category(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = PIL.Image.open(image)
    width, height = img.size 
    
    if width > height: 
        size = [width, 256]
    else: 
        size = [256, height]
        
    img.thumbnail(size)
    
    center = width/4, height/4 
    left = center[0]-(244/2) 
    top = center[1]-(244/2)
    right = center[0]+(244/2)
    bottom = center[1]+(244/2)
    
    img = img.crop((left, top, right, bottom))
    img = np.array(img)/255 
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = ((img - mean) / std)
    
    img = np.transpose(img, (2, 0, 1))
    
    return img 

def predict(image_path, model, gpu, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.to('cuda')
    
    img_torch = process_image(image_path)
    img_torch = torch.from_numpy(img_torch)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda()) 
        
    probability = F.softmax(output.data,dim=1) 
    
    probs = np.array(probability.topk(topk)[0][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()} 
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, top_classes

def main(): 
    print("start")
    arg = arg_parse()
    gpu = arg.gpu
    model = load_checkpoint(arg.checkpoint)
    cat_to_name = load_category(arg.category_names)
    
    img_path = arg.image_path
    probs, classes = predict(img_path, model, gpu, int(arg.top_k))
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)
    
    print(labels)
    print(probability)
    
    i=0 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 

if __name__ == "__main__":
    main()