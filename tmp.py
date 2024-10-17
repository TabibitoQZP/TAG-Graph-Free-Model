import simplejson as json

if __name__ == '__main__':
    with open('processed/pubmed/text.json', 'r') as js:
        textList = json.load(js)

    for t in textList:
        st = t.split('\n')
        if len(st) != 2:
            print(len(st))
