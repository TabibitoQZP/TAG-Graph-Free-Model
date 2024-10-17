import os
import random
import simplejson as json
from transformers import T5Tokenizer, T5ForSequenceClassification, T5EncoderModel

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class ClassificationModel(nn.Module):
    def __init__(self, classCount):
        super(ClassificationModel, self).__init__()
        self.baseModel = T5ForSequenceClassification.from_pretrained('google-t5/t5-small', num_labels=classCount)
        # self.baseModel = T5EncoderModel.from_pretrained('google-t5/t5-small')
        # for param in self.baseModel.parameters():
        #     param.requires_grad = False
        # self.linear = nn.Linear(512, classCount)

    def forward(self, ids):
        # ret = self.baseModel(ids).last_hidden_state
        # unkOne = ret[:,0,:]
        # return self.linear(unkOne)
        return self.baseModel(ids).logits

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

def main(datasetName, labelCount):
    device = 'cuda:3'
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    model = ClassificationModel(labelCount).to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optm = optim.Adam(model.parameters(), lr=1e-4)

    dataRoot = os.path.join('processed', datasetName)
    neighbor, text, selectedIdx, testIdx = fewShotSplit(dataRoot, labelCount, 100)

    trainSet = TAGDdataset(neighbor, text, selectedIdx)
    trainDL = DataLoader(trainSet, batch_size=8, shuffle=True)

    testSet = TAGDdataset(neighbor, text, testIdx)
    testDL = DataLoader(testSet, batch_size=8)

    for i in range(100):
        model.train()
        tl = 0
        for bx, by in trainDL:
            optm.zero_grad()
            tknzed = tokenizer(bx, return_tensors="pt", padding=True)
            input_ids = tknzed.input_ids.to(device)
            by = by.to(device)
            ret = model(input_ids)
            l = loss(ret, by)
            l.backward()
            optm.step()
            tl += l
        print(tl)
        if i % 10 != 0:
            continue
        model.eval()
        acc = 0
        tot = 0
        for bx, by in testDL:
            input_ids = tokenizer(bx, return_tensors="pt", padding=True).input_ids.to(device)
            by = by.to(device)
            ret = model(input_ids)
            acc += (ret.argmax(1) == by).sum().item()
            tot += by.shape[0]
        print(acc / tot)


if __name__ == '__main__':
    # main('cora', 7)
    main('pubmed', 3)
    pass
