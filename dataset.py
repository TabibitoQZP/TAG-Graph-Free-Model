import os
import simplejson as json

from torch.utils.data import Dataset

def TAGDdataset(Dataset):
    def __init__(self, dataRoot):
        neighborPath = os.path.join(dataRoot, 'neighbor.json')
        textPath = os.path.join(dataRoot, 'text.json')

        with open(neighborPath, 'r') as js:
            self.neighbor = json.load(js)

        with open(textPath, 'r') as js:
            self.text = json.load(js)

    def __getitem__(self, idx):
        label = self.neighbor[idx]['label']
        text = self.text[idx]
        return text, label

    def __len__(self):
        return len(self.text)
