import os
from tagUtils import fewShotSplit, TAGDdataset, dataSplit
from model.BasicT5Classcification import BasicT5Classification
from model import TKNZ
from torch import optim
from torch.utils.data import DataLoader

def main(datasetName, labelCount):
    device = 'cuda:3'
    tokenizer = TKNZ(device)
    model = BasicT5Classification(labelCount).to(device)

    optm = optim.Adam(model.parameters(), lr=1e-4)

    dataRoot = os.path.join('processed', datasetName)
    neighbor, text, selectedIdx, testIdx = dataSplit(dataRoot)

    trainSet = TAGDdataset(neighbor, text, selectedIdx)
    trainDL = DataLoader(trainSet, batch_size=8, shuffle=True)

    testSet = TAGDdataset(neighbor, text, testIdx)
    testDL = DataLoader(testSet, batch_size=8)

    for i in range(100):
        model.train()
        tl = 0
        for bx, by in trainDL:
            optm.zero_grad()
            ids, msk = tokenizer(bx)
            by = by.to(device)
            l = model(ids, msk, by)[1]
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
            ids, msk = tokenizer(bx)
            by = by.to(device)
            ret = model(ids, msk)[0]
            acc += (ret.argmax(1) == by).sum().item()
            tot += by.shape[0]
        print(acc / tot)

if __name__ == '__main__':
    main('cora', 7)
