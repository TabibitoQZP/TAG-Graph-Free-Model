import os
import random
import simplejson as json
from torch.utils.data import Dataset

def dataSplit(dataRoot):
    def loadData(pth):
        with open(pth, 'r') as js:
            ld = json.load(js)
        return ld
    neighborPath = os.path.join(dataRoot, 'neighbor.json')
    textPath = os.path.join(dataRoot, 'text.json')
    trainPath = os.path.join(dataRoot, 'train.json')
    validPath = os.path.join(dataRoot, 'valid.json')
    testPath = os.path.join(dataRoot, 'test.json')
    neighbor = loadData(neighborPath)
    text = loadData(textPath)
    train = loadData(trainPath)
    test = loadData(testPath)
    return neighbor, text, train, test

def fewShotSplit(dataRoot, labelCount, examplePerClass):
    neighborPath = os.path.join(dataRoot, 'neighbor.json')
    textPath = os.path.join(dataRoot, 'text.json')
    with open(neighborPath, 'r') as js:
        neighbor = json.load(js)

    with open(textPath, 'r') as js:
        text = json.load(js)

    sz = len(text)
    labelIdx = [list() for _ in range(labelCount)]

    for i in range(sz):
        labelIdx[neighbor[i]['label']].append(i)

    selectedIdx = []
    for li in labelIdx:
        selectedIdx.extend(random.sample(li, examplePerClass))

    testIdx = [i for i in range(sz) if i not in selectedIdx]
    return neighbor, text, selectedIdx, testIdx

class TAGDdataset(Dataset):
    def __init__(self, neighbor, text, idxList):
        self.neighbor = neighbor
        self.text = text
        self.idxList = idxList

    def __getitem__(self, idx):
        idx = self.idxList[idx]
        label = self.neighbor[idx]['label']
        neighbor = self.neighbor[idx]['neighbor']
        selectedNeighbor = random.sample(neighbor, min(len(neighbor), 4))
        return self.text[idx], label
        text = self.text[idx].split('\n')[0].strip()
        text = f'<unk> is `{text}`.'

        snStr = []
        for idx in selectedNeighbor:
            snStr.append(self.text[idx].split("\n")[0].strip())
            snStr[-1] = f'`{snStr[-1]}`'
        text = text + ' It is link to ' + ', '.join(snStr) + '.'
        return text, label

    def __len__(self):
        return len(self.idxList)
