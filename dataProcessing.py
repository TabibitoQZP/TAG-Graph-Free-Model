import os
import simplejson as json


from tagUtils.load_cora import get_cora_casestudy, get_raw_text_cora
from tagUtils.load_pubmed import get_pubmed_casestudy, get_raw_text_pubmed

def processingData(data, text, datasetName):
    nodeCount = data.x.shape[0]
    edges = data.edge_index
    labels = data.y.tolist()
    # 生成配置文件
    dstRoot = f'processed/{datasetName}'
    os.makedirs(dstRoot, exist_ok=True)
    neighborPath = os.path.join(dstRoot, 'neighbor.json')
    textPath = os.path.join(dstRoot, 'text.json')
    with open(textPath, 'w') as js:
        json.dump(text, js, indent=2)


    neighbor = []
    for i in range(nodeCount):
        neighbor.append({
            'label': labels[i],
            'neighbor': []
        })
    for i in range(edges.shape[1]):
        item = edges[:,i].tolist()
        neighbor[item[0]]['neighbor'].append(item[1])
        neighbor[item[1]]['neighbor'].append(item[0])
    for item in neighbor:
        item['neighbor'].sort()
    with open(neighborPath, 'w') as js:
        json.dump(neighbor, js, indent=2)


if __name__ == '__main__':
    data, text = get_raw_text_cora(use_text=True)
    processingData(data, text, 'cora')
    data, text = get_raw_text_pubmed(use_text=True)
    processingData(data, text, 'pubmed')
