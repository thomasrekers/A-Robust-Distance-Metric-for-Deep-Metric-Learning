#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""
import os
from torch.utils.data import Dataset
#from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import torchvision.transforms as transforms

import torch.utils.data as data

import os
import os.path


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def get_transform():
    # Data transforms
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    transform_list = []
    transform_list.append(transforms.Resize((227,227)))
    #	transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=stdv))

    return transforms.Compose(transform_list)


# rename: single_dataset.py
class MyDataset(Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.
    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """
    def __init__(self,dataroot,phase,image_list_file,transform=get_transform):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
          # ./chest_Xray/train   or ./chest_Xray/test
        
    
        self.dir_A = os.path.join(dataroot, phase) # phase = 'train', 'test'  
        self.paths = sorted(make_dataset(self.dir_A))#, max_dataset_size))
                 
        images = []
        labels = []
        with open(image_list_file,'r') as f:
            for line in f: 
                tmp = line.split()
                
                image = tmp[0]
                label = int(tmp[1])
                images.append(image)
                labels.append(label)
                
        self.images = images
        self.labels = labels
        self.transform = transform()
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            image and its label
        """
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
       
        
       
        _,tmp_name = os.path.split(path)
        index = self.images.index(tmp_name)
        label = [self.labels[index]]
        
        
        
        image = self.transform(img)
        
        
        
    
        return image,torch.FloatTensor(label)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)

