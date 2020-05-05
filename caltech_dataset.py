from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import pandas as pd

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    def __init__(self, root, split = 'train', transform = None, target_transform = None):
        
        super(Caltech, self).__init__(root, transform = transform, target_transform = target_transform)

        self.split = split
        
        avoid_folder = "BACKGROUND_Google"
        
        self._categories = sorted(os.listdir(root))
        self._categories.remove(avoid_folder)
        
        self._labels_id = {label : i for i, label in enumerate(self._categories)}
        
        self._data = pd.DataFrame()

        with open(root + "/../" + split + ".txt", "r") as f:
            content = f.readlines()
        
        labels = []
        records = []
        
        for i, row in enumerate(content):
            label, image_name = row[:-1].split("/")
            if label != avoid_folder:
                labels.append(self._labels_id[label])
                records.append(pil_loader(root + "/" + row[:-1]))

        self._data['image'] = records
        self._data['label'] = labels
    
    
    def getCategories(self):
        return self._categories;
    
    
    def getLabel(self, label):
        return self._labels_id[label];
    
    
    def getData(self):
        return self._data;
    
    
    def __getitem__(self, index):

        image, label = self._data.loc[index]['image'], self._data.loc[index]['label']

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    
    def __len__(self):
        return len(self._data)
