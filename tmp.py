import os
from tqdm import tqdm
import simplejson as json
from uuid import uuid4
from tagUtils import fewShotSplit, TAGDdataset, dataSplit
from model.BasicT5Classcification import BasicT5Classification
from model.MaskBasedClassification import MaskBasedClassification
from model import TKNZ
from torch import optim
from torch.utils.data import DataLoader

def main(datasetName, labelCount, batchSize=8, lr=1e-4, modelType='mask', fewShotNum=None, device='cuda:3'):
    os.makedirs('result', exist_ok=True)
    saveDict = {
            'datasetName': datasetName,
            'classNums': labelCount,
            'batchSize': batchSize,
            'lr': lr,
            'modelType': modelType,
            'fewShotNum': fewShotNum,
        }
    fname = '_'.join([str(item) for item in saveDict.values()]) + '.json'
    fPath = os.path.join('result', fname)
    if os.path.isfile(fPath):
        return None
    tokenizer = TKNZ(device)
    if modelType == 'mask':
        model = MaskBasedClassification(labelCount).to(device)
    else:
        model = BasicT5Classification(labelCount).to(device)

    optm = optim.Adam(model.parameters(), lr=lr)

    dataRoot = os.path.join('processed', datasetName)
    if fewShotNum is None:
        neighbor, text, selectedIdx, testIdx = dataSplit(dataRoot)
    else:
        neighbor, text, selectedIdx, testIdx = fewShotSplit(dataRoot, labelCount, fewShotNum)
    if datasetName == 'pubmed':
        # 这个数据集很大很慢, 最好是只拿一部分测试
        testIdx = testIdx[:1000]

    if modelType == 'mask':
        trainSet = TAGDdataset(neighbor, text, selectedIdx, True)
    else:
        trainSet = TAGDdataset(neighbor, text, selectedIdx, False)
    trainDL = DataLoader(trainSet, batch_size=1, shuffle=True) # batchsize统一设置为1, 防止OOM, batchSize通过设计zero_grad次数实现

    if modelType == 'mask':
        testSet = TAGDdataset(neighbor, text, testIdx, True)
    else:
        testSet = TAGDdataset(neighbor, text, testIdx, False)
    testDL = DataLoader(testSet, batch_size=1)

    cnt = 0
    maxRate = 0
    turns = []
    rates = []
    turnNum = 1001
    checkNum = 10
    if fewShotNum is None:
        turnNum = 101
    for i in tqdm(range(1, turnNum)):
        model.train()
        tl = 0
        for bx, by in trainDL:
            ids, msk = tokenizer(bx)
            by = by.to(device)
            l = model(ids, msk, by)[1]
            l.backward()
            cnt += 1
            if cnt % batchSize == 0:
                optm.step()
                optm.zero_grad()
                cnt = 0
            tl += l
        turns.append(tl.item())
        if i % checkNum != 0:
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
        rate = acc / tot
        rates.append(rate)
        if rate > maxRate:
            maxRate = rate

    saveDict.update({
            'device': device,
            'maxRate': maxRate,
            'rates': rates,
            'turns': turns
    })
    with open(fPath, 'w') as js:
        json.dump(saveDict, js, indent=2)

if __name__ == '__main__':
    datasetWithCount = [('cora', 7), ('pubmed', 3)]
    models = ['mask', 'other']
    fewShotNums = [3, 5, 16, None]
    for dc in datasetWithCount:
        for m in models:
            for fn in fewShotNums:
                try:
                    main(dc[0], dc[1], 8, fewShotNum=fn, modelType=m, device='cuda:7')
                except:
                    pass
