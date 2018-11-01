import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random
import math
import os
from tensorboardX import SummaryWriter


torch.manual_seed(12)
random.seed(12)

batch_size = 50

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}

def conv3x3(in_planes, out_planes, cardinality, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=cardinality, bias=False)


class ResNextBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, cardinality=32, stride=1, downsample=False):
        super(ResNextBasicBlock, self).__init__()
        if planes // cardinality == 0:
            cardinality = 1
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, inplanes, cardinality, stride = stride)
        self.bn2 = nn.BatchNorm2d(inplanes)        
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        self.cardinality = cardinality
        
        self.added_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.added_bn = nn.BatchNorm2d(planes)
        

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
 
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample or self.stride != 1:
            residual = self.added_conv(residual)
            residual = self.added_bn(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNext(nn.Module):
    def __init__(self, block, layers, cardinality=32, num_classes=10):
        self.inplanes = 64
        self.cardinality = cardinality
        super(ResNext, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer1(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer1(self, block, planes, number_of_blocks, stride=1):            
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, downsample=True))
        for i in range(1, number_of_blocks):
             layers.append(block(planes, planes, self.cardinality))
            
        return nn.Sequential(*layers)

    
    def _make_layer(self, block, planes, number_of_blocks, stride):            
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride=2))
        self.inplanes *= 2
        for i in range(1, number_of_blocks):
             layers.append(block(planes, planes, self.cardinality, stride=1))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext(cardinality=32, path_to_pretrained_model=None, **kwargs):
    model = ResNext(ResNextBasicBlock, [3, 4, 6, 3], cardinality, **kwargs)
    if path_to_pretrained_model is not None:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # model.load_state_dict(torch.load(path_to_pretrained_model))
    return model


class Trainer:
    def __init__(self, model, train_loader, epochs, learning_rate, output_dir):
        self.model = model
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.make_dir(output_dir)
        self.output_dir = output_dir
    
    def make_dir(self, path):
        try:
            os.mkdir(path)
        except OSError:
            pass
    
    def train_model(self):
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
  
        writer = SummaryWriter(self.output_dir)
        for index in range(self.epochs):
            for batch_index, (images, labels) in enumerate(self.train_loader):
                images = Variable(images.float())
                labels = Variable(labels)
                outputs = self.model(images)
                loss = criterion(outputs, labels)   
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print('loss', loss.item(), index)
                writer.add_scalar('loss', loss, index)       
        
        writer.close()
